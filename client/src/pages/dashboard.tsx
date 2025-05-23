import React, { useState, useEffect, useRef } from 'react';
import { useAuth } from '@/hooks/use-auth';
import { useToast } from '@/hooks/use-toast';
import { useTheme } from '@/components/providers/theme-provider';
import { useQuery, useMutation } from '@tanstack/react-query';
import { apiRequest, queryClient } from '@/lib/queryClient';
import { connectWebSocket, sendChatMessage, registerMessageHandler } from '@/lib/websocket';
import { AIMessage, WebSocketMessage } from '@/lib/types';
import { 
  Loader2, 
  Code, 
  Cpu, 
  GitBranch, 
  Play, 
  Pause, 
  RefreshCw, 
  FileText, 
  ArrowRight, 
  Brain,
  Settings,
  BarChart3,
  Search,
  MessageSquare,
  Plus,
  X,
  Check,
  ChevronRight,
  Terminal,
  Eye,
  Lightbulb,
  AlertTriangle
} from 'lucide-react';
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import * as z from "zod";
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";

import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Textarea } from '@/components/ui/textarea';
import { Input } from '@/components/ui/input';

export default function Dashboard() {
  const { user, logoutMutation } = useAuth();
  const { toast } = useToast();
  const { theme, setTheme } = useTheme();
  const [activeTab, setActiveTab] = useState('overview');
  const [chatMessages, setChatMessages] = useState<AIMessage[]>([
    {
      conversationId: 'default',
      role: 'assistant',
      content: 'Hello! I\'m Seren, your advanced AI assistant. I\'m designed to help you with a wide range of tasks including coding, reasoning, problem-solving and more. How can I assist you today?'
    }
  ]);
  const [userInput, setUserInput] = useState('');
  const [selectedModel, setSelectedModel] = useState<'qwen' | 'olympic' | 'hybrid'>('hybrid');
  const [isLoadingResponse, setIsLoadingResponse] = useState(false);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const conversationId = useRef<string>('default-' + Date.now());
  
  // Handle WebSocket connection for real-time chat
  useEffect(() => {
    if (!user) return;
    
    // Connect to WebSocket when component mounts
    const socket = connectWebSocket(user.id);
    
    // Register message handler for incoming messages
    const unregister = registerMessageHandler((data) => {
      if (data.type === 'new-message' && data.message) {
        // Add new message to chat
        setChatMessages(prev => [...prev, data.message]);
        
        // If it's an assistant response, we're no longer loading
        if (data.message.role === 'assistant') {
          setIsLoadingResponse(false);
        }
      }
    });
    
    return () => {
      // Clean up when component unmounts
      unregister();
    };
  }, [user]);
  
  // Scroll to bottom when new messages are added
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [chatMessages]);
  
  // Get models status
  const { data: modelStatus, isLoading: modelsLoading } = useQuery({
    queryKey: ['/api/ai/models/status'],
    refetchInterval: 10000,
    staleTime: 5000,
    enabled: false, // Temporarily disable this query until API is ready
  });

  // Get system settings
  const { data: systemSettings, isLoading: settingsLoading } = useQuery({
    queryKey: ['/api/settings'],
    staleTime: 60000,
    enabled: false, // Temporarily disable this query until API is ready
  });
  
  // Get chat history
  const { data: chatHistory } = useQuery({
    queryKey: ['/api/messages', conversationId.current],
    enabled: activeTab === 'chat',
    onSuccess: (data) => {
      if (data && Array.isArray(data) && data.length > 0) {
        setChatMessages(data);
      }
    },
    onError: (error) => {
      // If there's an error, we'll keep using the existing messages
      console.error('Failed to fetch chat history:', error);
    }
  });

  // Handle sending messages
  const sendMessage = async () => {
    if (!userInput.trim() || isLoadingResponse) return;
    
    // Create user message
    const userMessage: AIMessage = {
      conversationId: conversationId.current,
      role: 'user',
      content: userInput,
      model: selectedModel, // Use the actual model names as selected
      userId: user?.id,
      metadata: { timestamp: new Date().toISOString() }
    };
    
    // Add user message to state
    setChatMessages(prev => [...prev, userMessage]);
    
    // Clear input
    setUserInput('');
    
    // Show loading state
    setIsLoadingResponse(true);
    
    try {
      // Send through WebSocket for real-time handling
      const success = sendChatMessage(userMessage);
      
      // If WebSocket fails or isn't connected, fall back to REST API
      if (!success) {
        const response = await apiRequest('POST', '/api/messages', userMessage);
        const assistantMessage = await response.json();
        
        // Add assistant response to chat
        setChatMessages(prev => [...prev, assistantMessage]);
        setIsLoadingResponse(false);
      }
    } catch (error) {
      console.error('Failed to send message:', error);
      toast({
        title: 'Failed to send message',
        description: 'Please try again later.',
        variant: 'destructive'
      });
      setIsLoadingResponse(false);
    }
  };

  // Handle logout
  const handleLogout = () => {
    logoutMutation.mutate();
  };

  // Simulation data for models (until API is ready)
  const simulatedModelStatus = {
    qwen: {
      status: "ready",
      uptime: "2h 15m",
      memory: "4.2 GB",
      supportedTasks: ["text-generation", "reasoning", "code-generation"]
    },
    olympicCoder: {
      status: "ready",
      uptime: "2h 15m",
      memory: "4.0 GB",
      supportedTasks: ["code-generation", "code-optimization", "bug-fixing"]
    }
  };

  // Loading state
  if (!user) {
    return (
      <div className="h-screen flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }
  
  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b">
        <div className="container mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Brain className="h-8 w-8 text-primary" />
            <h1 className="text-xl font-bold">Seren AI</h1>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-sm text-muted-foreground">
              Logged in as <span className="font-medium text-foreground">{user.username}</span>
              {user.isAdmin && <Badge className="ml-2 bg-primary">Admin</Badge>}
            </span>
            <Button variant="outline" size="sm" onClick={handleLogout} disabled={logoutMutation.isPending}>
              {logoutMutation.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                "Logout"
              )}
            </Button>
          </div>
        </div>
      </header>

      {/* Main content */}
      <div className="container mx-auto px-4 py-8">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
            <p className="text-muted-foreground">
              Welcome to Seren, your advanced AI development platform
            </p>
          </div>
        </div>

        <Tabs defaultValue="overview" value={activeTab} onValueChange={setActiveTab} className="space-y-4">
          <TabsList>
            <TabsTrigger value="overview">
              <BarChart3 className="h-4 w-4 mr-2" />
              Overview
            </TabsTrigger>
            <TabsTrigger value="projects">
              <GitBranch className="h-4 w-4 mr-2" />
              Projects
            </TabsTrigger>
            <TabsTrigger value="models">
              <Brain className="h-4 w-4 mr-2" />
              AI Models
            </TabsTrigger>
            <TabsTrigger value="chat">
              <MessageSquare className="h-4 w-4 mr-2" />
              AI Chat
            </TabsTrigger>
            <TabsTrigger value="settings">
              <Settings className="h-4 w-4 mr-2" />
              Settings
            </TabsTrigger>
          </TabsList>
          
          {/* Projects Tab */}
          <TabsContent value="projects" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="md:col-span-2">
                <Card className="h-full">
                  <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                    <div>
                      <CardTitle>Your Projects</CardTitle>
                      <CardDescription>
                        View and manage your AI-assisted software projects
                      </CardDescription>
                    </div>
                    <Button size="sm" className="gap-1">
                      <Plus className="h-4 w-4" />
                      New Project
                    </Button>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {/* Sample projects list */}
                      {[
                        {
                          id: 1,
                          name: "E-commerce Platform",
                          description: "Modern React e-commerce site with payment integration",
                          status: "In Progress",
                          lastUpdated: "2 hours ago",
                          progress: 65
                        },
                        {
                          id: 2,
                          name: "Personal Finance API",
                          description: "RESTful API with authentication and data visualization",
                          status: "Completed",
                          lastUpdated: "2 days ago",
                          progress: 100
                        },
                        {
                          id: 3,
                          name: "Task Management App",
                          description: "Mobile-friendly task organizer with team collaboration",
                          status: "Planning",
                          lastUpdated: "4 hours ago",
                          progress: 20
                        }
                      ].map((project) => (
                        <div key={project.id} className="border rounded-lg p-4 space-y-3 hover:bg-accent/50 transition-colors cursor-pointer">
                          <div className="flex items-center justify-between">
                            <div>
                              <h3 className="text-base font-medium">{project.name}</h3>
                              <p className="text-sm text-muted-foreground">{project.description}</p>
                            </div>
                            <ChevronRight className="h-5 w-5 text-muted-foreground" />
                          </div>
                          
                          <div className="space-y-2">
                            <div className="flex justify-between text-sm">
                              <span className="text-muted-foreground">Status:</span>
                              <Badge variant={
                                project.status === "Completed" ? "outline" : 
                                project.status === "In Progress" ? "secondary" : 
                                "default"
                              }>
                                {project.status}
                              </Badge>
                            </div>
                            
                            <div className="flex justify-between text-sm">
                              <span className="text-muted-foreground">Last updated:</span>
                              <span>{project.lastUpdated}</span>
                            </div>
                            
                            <div className="space-y-1">
                              <div className="flex justify-between text-sm">
                                <span className="text-muted-foreground">Progress:</span>
                                <span>{project.progress}%</span>
                              </div>
                              <Progress value={project.progress} />
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
              
              <div className="md:col-span-1">
                <Card>
                  <CardHeader>
                    <CardTitle>Create New Project</CardTitle>
                    <CardDescription>
                      Define requirements to have AI generate your project
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <label className="text-sm font-medium" htmlFor="project-name">
                        Project Name
                      </label>
                      <Input id="project-name" placeholder="Enter project name" />
                    </div>
                    
                    <div className="space-y-2">
                      <label className="text-sm font-medium" htmlFor="project-description">
                        Description & Requirements
                      </label>
                      <Textarea 
                        id="project-description" 
                        placeholder="Describe what you want the AI to build..." 
                        className="min-h-[120px]"
                      />
                      <p className="text-xs text-muted-foreground">
                        Be as detailed as possible about features, technologies, and goals.
                      </p>
                    </div>
                    
                    <div className="space-y-2">
                      <label className="text-sm font-medium">
                        Preferred Language
                      </label>
                      <Select defaultValue="javascript">
                        <SelectTrigger>
                          <SelectValue placeholder="Select language" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="javascript">JavaScript/TypeScript</SelectItem>
                          <SelectItem value="python">Python</SelectItem>
                          <SelectItem value="java">Java</SelectItem>
                          <SelectItem value="csharp">C#</SelectItem>
                          <SelectItem value="go">Go</SelectItem>
                          <SelectItem value="rust">Rust</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    
                    <div className="space-y-2">
                      <label className="text-sm font-medium">
                        Preferred Framework
                      </label>
                      <Select defaultValue="react">
                        <SelectTrigger>
                          <SelectValue placeholder="Select framework" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="react">React</SelectItem>
                          <SelectItem value="vue">Vue</SelectItem>
                          <SelectItem value="angular">Angular</SelectItem>
                          <SelectItem value="nextjs">Next.js</SelectItem>
                          <SelectItem value="django">Django</SelectItem>
                          <SelectItem value="flask">Flask</SelectItem>
                          <SelectItem value="express">Express.js</SelectItem>
                          <SelectItem value="spring">Spring Boot</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <Switch id="advanced-features" />
                      <label className="text-sm font-medium" htmlFor="advanced-features">
                        Include advanced AI features
                      </label>
                    </div>
                  </CardContent>
                  <CardFooter>
                    <Button className="w-full gap-2">
                      <Lightbulb className="h-4 w-4" />
                      Generate Project
                    </Button>
                  </CardFooter>
                </Card>
              </div>
            </div>
          </TabsContent>
          
          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">Models Status</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">2/2</div>
                  <p className="text-xs text-muted-foreground">
                    Models online and operational
                  </p>
                  <div className="mt-4 h-1 w-full bg-secondary">
                    <div className="h-1 bg-primary" style={{ width: '100%' }} />
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">Knowledge Base</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">24.5k</div>
                  <p className="text-xs text-muted-foreground">
                    Entries in knowledge library
                  </p>
                  <div className="mt-4 h-1 w-full bg-secondary">
                    <div className="h-1 bg-primary" style={{ width: '80%' }} />
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">System Health</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">98.5%</div>
                  <p className="text-xs text-muted-foreground">
                    System operational efficiency
                  </p>
                  <div className="mt-4 h-1 w-full bg-secondary">
                    <div className="h-1 bg-primary" style={{ width: '98.5%' }} />
                  </div>
                </CardContent>
              </Card>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Card className="col-span-1">
                <CardHeader>
                  <CardTitle>Recent Activity</CardTitle>
                  <CardDescription>
                    Latest system operations and events
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-[300px]">
                    <div className="space-y-4">
                      {Array.from({ length: 5 }).map((_, i) => (
                        <div key={i} className="flex items-start gap-4">
                          <div className="rounded-full p-2 bg-primary/10">
                            <Brain className="h-4 w-4 text-primary" />
                          </div>
                          <div className="space-y-1">
                            <p className="text-sm font-medium">AI Model Training Complete</p>
                            <p className="text-xs text-muted-foreground">
                              Neurosymbolic module trained on new data
                            </p>
                            <p className="text-xs text-muted-foreground">
                              {new Date(Date.now() - i * 3600000).toLocaleString()}
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
              
              <Card className="col-span-1">
                <CardHeader>
                  <CardTitle>System Architecture</CardTitle>
                  <CardDescription>
                    Current system components and integrations
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Brain className="h-5 w-5 text-primary" />
                        <span className="font-medium">AI Core</span>
                      </div>
                      <Badge variant="outline" className="text-green-600 bg-green-50">Active</Badge>
                    </div>
                    <Separator />
                    
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <ArrowRight className="h-5 w-5 text-primary" />
                        <span className="font-medium">Neuro-Symbolic Reasoning</span>
                      </div>
                      <Badge variant="outline" className="text-green-600 bg-green-50">Active</Badge>
                    </div>
                    <Separator />
                    
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <RefreshCw className="h-5 w-5 text-primary" />
                        <span className="font-medium">Metacognitive System</span>
                      </div>
                      <Badge variant="outline" className="text-green-600 bg-green-50">Active</Badge>
                    </div>
                    <Separator />
                    
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <FileText className="h-5 w-5 text-primary" />
                        <span className="font-medium">Knowledge Library</span>
                      </div>
                      <Badge variant="outline" className="text-green-600 bg-green-50">Active</Badge>
                    </div>
                    <Separator />
                    
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Code className="h-5 w-5 text-primary" />
                        <span className="font-medium">Code Generation</span>
                      </div>
                      <Badge variant="outline" className="text-green-600 bg-green-50">Active</Badge>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
          
          {/* AI Models Tab */}
          <TabsContent value="models" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Card>
                <CardHeader>
                  <CardTitle>Qwen2.5-7b-omni</CardTitle>
                  <CardDescription>
                    Advanced reasoning and general-purpose AI model
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm font-medium">Status</p>
                      <Badge variant="outline" className="mt-1 text-green-600 bg-green-50">
                        {simulatedModelStatus.qwen.status}
                      </Badge>
                    </div>
                    <div>
                      <p className="text-sm font-medium">Uptime</p>
                      <p className="text-sm">{simulatedModelStatus.qwen.uptime}</p>
                    </div>
                    <div>
                      <p className="text-sm font-medium">Memory Usage</p>
                      <p className="text-sm">{simulatedModelStatus.qwen.memory}</p>
                    </div>
                    <div>
                      <p className="text-sm font-medium">Model Type</p>
                      <p className="text-sm">Qwen 2.5 Omni</p>
                    </div>
                  </div>
                  
                  <div>
                    <p className="text-sm font-medium mb-2">Supported Tasks</p>
                    <div className="flex flex-wrap gap-2">
                      {simulatedModelStatus.qwen.supportedTasks.map((task, i) => (
                        <Badge key={i} variant="secondary">{task}</Badge>
                      ))}
                    </div>
                  </div>
                  
                  <Separator />
                  
                  <div className="flex justify-between">
                    <Button variant="outline" size="sm">
                      View Logs
                    </Button>
                    <Button variant="outline" size="sm">
                      <RefreshCw className="h-4 w-4 mr-2" />
                      Refresh Status
                    </Button>
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader>
                  <CardTitle>OlympicCoder-7B</CardTitle>
                  <CardDescription>
                    Specialized code generation and software development model
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm font-medium">Status</p>
                      <Badge variant="outline" className="mt-1 text-green-600 bg-green-50">
                        {simulatedModelStatus.olympicCoder.status}
                      </Badge>
                    </div>
                    <div>
                      <p className="text-sm font-medium">Uptime</p>
                      <p className="text-sm">{simulatedModelStatus.olympicCoder.uptime}</p>
                    </div>
                    <div>
                      <p className="text-sm font-medium">Memory Usage</p>
                      <p className="text-sm">{simulatedModelStatus.olympicCoder.memory}</p>
                    </div>
                    <div>
                      <p className="text-sm font-medium">Model Type</p>
                      <p className="text-sm">Olympic Coder</p>
                    </div>
                  </div>
                  
                  <div>
                    <p className="text-sm font-medium mb-2">Supported Tasks</p>
                    <div className="flex flex-wrap gap-2">
                      {simulatedModelStatus.olympicCoder.supportedTasks.map((task, i) => (
                        <Badge key={i} variant="secondary">{task}</Badge>
                      ))}
                    </div>
                  </div>
                  
                  <Separator />
                  
                  <div className="flex justify-between">
                    <Button variant="outline" size="sm">
                      View Logs
                    </Button>
                    <Button variant="outline" size="sm">
                      <RefreshCw className="h-4 w-4 mr-2" />
                      Refresh Status
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </div>
            
            <Card>
              <CardHeader>
                <CardTitle>Model Integration</CardTitle>
                <CardDescription>
                  Control how the AI models collaborate and interact
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="space-y-2">
                      <div className="font-medium">Collaboration Mode</div>
                      <Select defaultValue="hybrid">
                        <SelectTrigger>
                          <SelectValue placeholder="Select collaboration mode" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="collaborative">Collaborative</SelectItem>
                          <SelectItem value="specialized">Specialized</SelectItem>
                          <SelectItem value="competitive">Competitive</SelectItem>
                          <SelectItem value="hybrid">Hybrid</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    
                    <div className="space-y-2">
                      <div className="font-medium">Primary Model</div>
                      <Select defaultValue="both">
                        <SelectTrigger>
                          <SelectValue placeholder="Select primary model" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="qwen">Qwen2.5-7b-omni</SelectItem>
                          <SelectItem value="olympicCoder">OlympicCoder-7B</SelectItem>
                          <SelectItem value="both">Both (Equal)</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    
                    <div className="space-y-2">
                      <div className="font-medium">Processing Priority</div>
                      <Select defaultValue="balanced">
                        <SelectTrigger>
                          <SelectValue placeholder="Select processing priority" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="speed">Speed</SelectItem>
                          <SelectItem value="quality">Quality</SelectItem>
                          <SelectItem value="balanced">Balanced</SelectItem>
                          <SelectItem value="efficiency">Efficiency</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                  
                  <Separator />
                  
                  <div className="space-y-2">
                    <div className="font-medium">Reasoning Depth</div>
                    <div className="flex items-center space-x-2">
                      <Slider defaultValue={[7]} max={10} step={1} />
                      <span className="w-12 text-center">7/10</span>
                    </div>
                  </div>
                  
                  <Separator />
                  
                  <Button>Save Configuration</Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          
          {/* AI Chat Tab */}
          <TabsContent value="chat" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <div className="md:col-span-3">
                <Card className="h-[650px] flex flex-col">
                  <CardHeader className="border-b pb-3">
                    <div className="flex items-center justify-between">
                      <div>
                        <CardTitle>AI Chat Assistant</CardTitle>
                        <CardDescription>
                          Interact with the Seren AI core through natural language
                        </CardDescription>
                      </div>
                      <Select value={selectedModel} onValueChange={(val) => setSelectedModel(val as any)}>
                        <SelectTrigger className="w-[180px]">
                          <SelectValue placeholder="Select model" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="qwen">Qwen 2.5 Omni</SelectItem>
                          <SelectItem value="olympic">Olympic Coder</SelectItem>
                          <SelectItem value="hybrid">Hybrid (Default)</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </CardHeader>
                  <CardContent className="flex-1 flex flex-col overflow-hidden p-0">
                    <ScrollArea className="flex-1 px-4 py-6" ref={chatContainerRef}>
                      <div className="space-y-6 mb-4">
                        {chatMessages.map((message, index) => (
                          <div key={index} className={`flex items-start gap-3 ${message.role === 'user' ? 'justify-end' : ''}`}>
                            {message.role === 'assistant' && (
                              <Brain className="h-8 w-8 mt-1 p-1 rounded-md bg-primary text-primary-foreground" />
                            )}
                            
                            <div className={`${message.role === 'user' ? 'bg-primary/10' : 'bg-muted'} p-4 rounded-lg max-w-[85%]`}>
                              <p className="text-sm whitespace-pre-wrap">
                                {message.content}
                              </p>
                              {message.role === 'assistant' && message.content.includes('```') && (
                                <div className="mt-2 text-xs text-muted-foreground">
                                  <Button variant="ghost" size="sm" className="h-6 px-2">
                                    <Code className="h-3 w-3 mr-1" />
                                    <span>Copy code</span>
                                  </Button>
                                </div>
                              )}
                            </div>
                            
                            {message.role === 'user' && (
                              <div className="h-8 w-8 mt-1 bg-secondary rounded-full flex items-center justify-center">
                                <span className="text-sm font-medium">{user?.username?.charAt(0).toUpperCase() || 'U'}</span>
                              </div>
                            )}
                          </div>
                        ))}
                        
                        {isLoadingResponse && (
                          <div className="flex items-start gap-3">
                            <Brain className="h-8 w-8 mt-1 p-1 rounded-md bg-primary text-primary-foreground" />
                            <div className="bg-muted p-4 rounded-lg">
                              <div className="flex items-center space-x-2">
                                <div className="h-2 w-2 rounded-full bg-foreground animate-bounce"></div>
                                <div className="h-2 w-2 rounded-full bg-foreground animate-bounce delay-75"></div>
                                <div className="h-2 w-2 rounded-full bg-foreground animate-bounce delay-150"></div>
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    </ScrollArea>
                    
                    <div className="p-4 border-t bg-background">
                      <div className="flex gap-2">
                        <Textarea 
                          placeholder="Type your message..." 
                          className="flex-1 min-h-[80px]"
                          value={userInput}
                          onChange={(e) => setUserInput(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter' && !e.shiftKey) {
                              e.preventDefault();
                              sendMessage();
                            }
                          }}
                        />
                        <div className="flex flex-col gap-2">
                          <Button 
                            onClick={sendMessage} 
                            disabled={isLoadingResponse || !userInput.trim()}
                          >
                            {isLoadingResponse ? (
                              <Loader2 className="h-4 w-4 animate-spin" />
                            ) : (
                              <ArrowRight className="h-4 w-4" />
                            )}
                          </Button>
                          <Button variant="outline">
                            <Code className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                      <div className="flex justify-between items-center mt-2">
                        <div className="flex gap-2">
                          <Button 
                            variant="ghost" 
                            size="sm" 
                            className="h-8 px-2"
                            disabled={chatMessages.length <= 1 || isLoadingResponse}
                          >
                            <RefreshCw className="h-4 w-4 mr-1" />
                            <span className="text-xs">Regenerate</span>
                          </Button>
                          <Button variant="ghost" size="sm" className="h-8 px-2">
                            <FileText className="h-4 w-4 mr-1" />
                            <span className="text-xs">Upload File</span>
                          </Button>
                        </div>
                        <div className="text-xs text-muted-foreground">
                          {selectedModel === 'hybrid' ? 'Neuro-symbolic reasoning active' : 
                           selectedModel === 'qwen' ? 'Using Qwen 2.5 Omni model' : 
                           'Using Olympic Coder model'}
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
              
              <div className="md:col-span-1">
                <Card className="h-[650px] flex flex-col">
                  <CardHeader className="border-b pb-3">
                    <CardTitle className="text-sm">Chat Settings</CardTitle>
                  </CardHeader>
                  <CardContent className="flex-1 overflow-auto p-4">
                    <div className="space-y-6">
                      <div className="space-y-2">
                        <h4 className="text-sm font-medium">Model Selection</h4>
                        <div className="flex items-center justify-between">
                          <span className="text-sm">Primary Model</span>
                          <Select 
                            value={selectedModel}
                            onValueChange={(val) => setSelectedModel(val as any)}
                          >
                            <SelectTrigger className="w-[150px]">
                              <SelectValue placeholder="Select model" />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="qwen">Qwen 2.5</SelectItem>
                              <SelectItem value="olympic">Olympic Coder</SelectItem>
                              <SelectItem value="hybrid">Hybrid</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                      </div>
                      
                      <Separator />
                      
                      <div className="space-y-3">
                        <h4 className="text-sm font-medium">Advanced Settings</h4>
                        <div className="flex items-center justify-between">
                          <span className="text-sm">Temperature</span>
                          <div className="w-[120px]">
                            <Slider
                              defaultValue={[0.7]}
                              max={1.0}
                              step={0.1}
                            />
                          </div>
                        </div>
                        
                        <div className="flex items-center justify-between">
                          <span className="text-sm">Max Output</span>
                          <div className="w-[120px]">
                            <Slider
                              defaultValue={[2048]}
                              max={4096}
                              step={512}
                            />
                          </div>
                        </div>
                        
                        <div className="flex items-center justify-between">
                          <span className="text-sm">Creativity</span>
                          <div className="w-[120px]">
                            <Slider
                              defaultValue={[0.5]}
                              max={1.0}
                              step={0.1}
                            />
                          </div>
                        </div>
                      </div>
                      
                      <Separator />
                      
                      <div className="space-y-3">
                        <h4 className="text-sm font-medium">Features</h4>
                        <div className="flex items-center justify-between">
                          <span className="text-sm">Neuro-Symbolic</span>
                          <Switch defaultChecked />
                        </div>
                        
                        <div className="flex items-center justify-between">
                          <span className="text-sm">Knowledge Library</span>
                          <Switch defaultChecked />
                        </div>
                        
                        <div className="flex items-center justify-between">
                          <span className="text-sm">Code Generation</span>
                          <Switch defaultChecked />
                        </div>
                        
                        <div className="flex items-center justify-between">
                          <span className="text-sm">Reasoning Tracing</span>
                          <Switch />
                        </div>
                      </div>
                      
                      <Separator />
                      
                      <div className="space-y-2">
                        <h4 className="text-sm font-medium">Conversation</h4>
                        <Button 
                          variant="outline" 
                          size="sm" 
                          className="w-full"
                          onClick={() => {
                            setChatMessages([{
                              conversationId: 'default',
                              role: 'assistant',
                              content: 'Hello! I\'m Seren, your advanced AI assistant. I\'m designed to help you with a wide range of tasks including coding, reasoning, problem-solving and more. How can I assist you today?'
                            }]);
                            conversationId.current = 'default-' + Date.now();
                          }}
                        >
                          <X className="h-4 w-4 mr-2" />
                          Clear Conversation
                        </Button>
                        <Button variant="outline" size="sm" className="w-full">
                          <FileText className="h-4 w-4 mr-2" />
                          Export Chat
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>
          
          {/* Settings Tab */}
          <TabsContent value="settings" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>System Settings</CardTitle>
                <CardDescription>
                  Configure the AI system's behavior and parameters
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-6">
                  <div className="space-y-4">
                    <h3 className="text-lg font-medium">Display</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <div className="font-medium">Theme</div>
                        <Select value={theme} onValueChange={setTheme}>
                          <SelectTrigger>
                            <SelectValue placeholder="Select theme" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="dark">Dark (Default)</SelectItem>
                            <SelectItem value="light">Light</SelectItem>
                            <SelectItem value="system">System</SelectItem>
                          </SelectContent>
                        </Select>
                        <p className="text-sm text-muted-foreground">
                          The dark theme uses sleek black and purple styling for military-grade aesthetics.
                        </p>
                      </div>
                    </div>
                  </div>

                  <Separator />
                  
                  <div className="space-y-4">
                    <h3 className="text-lg font-medium">AI Core</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <div className="font-medium">Response Quality</div>
                        <Select defaultValue="balanced">
                          <SelectTrigger>
                            <SelectValue placeholder="Select quality level" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="fast">Fast (Lower Quality)</SelectItem>
                            <SelectItem value="balanced">Balanced</SelectItem>
                            <SelectItem value="high">High Quality (Slower)</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      
                      <div className="space-y-2">
                        <div className="font-medium">Memory Retention</div>
                        <Select defaultValue="medium">
                          <SelectTrigger>
                            <SelectValue placeholder="Select retention level" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="low">Low (Session-Only)</SelectItem>
                            <SelectItem value="medium">Medium (Days)</SelectItem>
                            <SelectItem value="high">High (Persistent)</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                  </div>
                </div>
                
                <Separator />
                
                <div className="space-y-4">
                  <h3 className="text-lg font-medium">Security</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <div className="font-medium">Quantum Encryption</div>
                      <div className="flex items-center space-x-2">
                        <Switch id="quantum-encryption" defaultChecked />
                        <label htmlFor="quantum-encryption">Enabled</label>
                      </div>
                    </div>
                    
                    <div className="space-y-2">
                      <div className="font-medium">Access Level</div>
                      <Select defaultValue="admin">
                        <SelectTrigger>
                          <SelectValue placeholder="Select access level" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="user">User</SelectItem>
                          <SelectItem value="developer">Developer</SelectItem>
                          <SelectItem value="admin">Admin</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                </div>
                
                <Separator />
                
                <div className="space-y-4">
                  <h3 className="text-lg font-medium">Advanced</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <div className="font-medium">Metacognitive System</div>
                      <div className="flex items-center space-x-2">
                        <Switch id="metacognitive" defaultChecked />
                        <label htmlFor="metacognitive">Enabled</label>
                      </div>
                    </div>
                    
                    <div className="space-y-2">
                      <div className="font-medium">Autonomous Evolution</div>
                      <div className="flex items-center space-x-2">
                        <Switch id="evolution" />
                        <label htmlFor="evolution">Disabled</label>
                      </div>
                    </div>
                  </div>
                </div>
                
                <Separator />
                
                <Button className="mr-2">Save Settings</Button>
                <Button variant="outline">Reset to Defaults</Button>
              </CardContent>
            </Card>
            
            {user.isAdmin && (
              <Card className="border-primary/50">
                <CardHeader>
                  <CardTitle>Admin Controls</CardTitle>
                  <CardDescription>
                    Advanced system management for administrators
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <Button variant="outline">System Backup</Button>
                    <Button variant="outline">Restore Checkpoint</Button>
                    <Button variant="outline">Export Logs</Button>
                    <Button variant="outline">Manage Users</Button>
                  </div>
                  
                  <Separator />
                  
                  <Button variant="destructive">
                    Emergency Shutdown
                  </Button>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );

  return (
    <div className="container mx-auto py-10">
      <h1 className="text-4xl font-bold mb-4">Seren AI Developer</h1>
      <p className="text-xl text-muted-foreground mb-8">
        Autonomous AI dev team with Qwen2.5-7b-omni and OlympicCoder-7B
      </p>

      <Tabs defaultValue="projects" value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="mb-8">
          <TabsTrigger value="projects">
            <GitBranch className="mr-2 h-4 w-4" />
            Autonomous Projects
          </TabsTrigger>
          <TabsTrigger value="direct">
            <Code className="mr-2 h-4 w-4" />
            Direct AI Coding
          </TabsTrigger>
          <TabsTrigger value="status">
            <Cpu className="mr-2 h-4 w-4" />
            System Status
          </TabsTrigger>
        </TabsList>

        {/* Autonomous Projects Tab */}
        <TabsContent value="projects" className="space-y-8">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <Card>
              <CardHeader>
                <CardTitle>Create New Project</CardTitle>
                <CardDescription>
                  Start a new autonomous software project
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Form {...projectForm}>
                  <form onSubmit={projectForm.handleSubmit(onProjectSubmit)} className="space-y-4">
                    <FormField
                      control={projectForm.control}
                      name="name"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Project Name</FormLabel>
                          <FormControl>
                            <Input placeholder="My Awesome Project" {...field} />
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                    
                    <FormField
                      control={projectForm.control}
                      name="description"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Description</FormLabel>
                          <FormControl>
                            <Input placeholder="A short description of your project" {...field} />
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />

                    <FormField
                      control={projectForm.control}
                      name="requirements"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Requirements</FormLabel>
                          <FormControl>
                            <Textarea 
                              placeholder="Detailed requirements for your project" 
                              className="min-h-[150px]" 
                              {...field} 
                            />
                          </FormControl>
                          <FormDescription>
                            Be as detailed as possible about what you want to build
                          </FormDescription>
                          <FormMessage />
                        </FormItem>
                      )}
                    />

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <FormField
                        control={projectForm.control}
                        name="language"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Language</FormLabel>
                            <FormControl>
                              <Input placeholder="e.g. JavaScript, Python" {...field} />
                            </FormControl>
                          </FormItem>
                        )}
                      />

                      <FormField
                        control={projectForm.control}
                        name="framework"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Framework</FormLabel>
                            <FormControl>
                              <Input placeholder="e.g. React, Django" {...field} />
                            </FormControl>
                          </FormItem>
                        )}
                      />
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <FormField
                        control={projectForm.control}
                        name="primaryModel"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Primary Model</FormLabel>
                            <Select 
                              onValueChange={field.onChange} 
                              defaultValue={field.value}
                            >
                              <FormControl>
                                <SelectTrigger>
                                  <SelectValue placeholder="Select a model" />
                                </SelectTrigger>
                              </FormControl>
                              <SelectContent>
                                <SelectItem value="hybrid">Hybrid (Both Models)</SelectItem>
                                <SelectItem value="qwen2.5-7b-omni">Qwen2.5-7b-omni</SelectItem>
                                <SelectItem value="olympiccoder-7b">OlympicCoder-7B</SelectItem>
                                <SelectItem value="openmanus">OpenManus (Advanced Agentic System)</SelectItem>
                              </SelectContent>
                            </Select>
                          </FormItem>
                        )}
                      />

                      <FormField
                        control={projectForm.control}
                        name="priorityLevel"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Priority</FormLabel>
                            <Select 
                              onValueChange={field.onChange} 
                              defaultValue={field.value}
                            >
                              <FormControl>
                                <SelectTrigger>
                                  <SelectValue placeholder="Select priority" />
                                </SelectTrigger>
                              </FormControl>
                              <SelectContent>
                                <SelectItem value="low">Low</SelectItem>
                                <SelectItem value="medium">Medium</SelectItem>
                                <SelectItem value="high">High</SelectItem>
                                <SelectItem value="critical">Critical</SelectItem>
                              </SelectContent>
                            </Select>
                          </FormItem>
                        )}
                      />
                    </div>

                    <Button 
                      type="submit" 
                      className="w-full"
                      disabled={createProjectMutation.isPending}
                    >
                      {createProjectMutation.isPending ? (
                        <>
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          Creating Project...
                        </>
                      ) : (
                        'Start Autonomous Project'
                      )}
                    </Button>
                  </form>
                </Form>
              </CardContent>
            </Card>

            <Tabs defaultValue="continuous" className="w-full">
              <TabsList className="w-full">
                <TabsTrigger value="continuous" className="flex-1">Continuous Execution</TabsTrigger>
                <TabsTrigger value="openmanus" className="flex-1">OpenManus Agentic</TabsTrigger>
              </TabsList>
              <TabsContent value="continuous">
                <Card>
                  <CardHeader>
                    <CardTitle>Your Projects</CardTitle>
                    <CardDescription>
                      Manage your autonomous software projects
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {projectsLoading ? (
                      <div className="flex items-center justify-center p-8">
                        <Loader2 className="h-8 w-8 animate-spin text-border" />
                      </div>
                    ) : projects && projects.projects && projects.projects.length > 0 ? (
                      <div className="space-y-4">
                        {projects.projects.map((project: any) => (
                          <Card 
                            key={project.id} 
                            className={`border-l-4 ${
                              selectedProject === project.id ? 'border-l-primary' : 'border-l-border'
                            } cursor-pointer hover:shadow-md transition-all`}
                            onClick={() => setSelectedProject(project.id)}
                          >
                            <CardContent className="p-4">
                              <div className="flex justify-between items-start">
                                <div>
                                  <h3 className="font-bold">{project.name}</h3>
                                  <p className="text-sm text-muted-foreground">{project.description}</p>
                                </div>
                                <div className="flex gap-2">
                                  <Badge className={getStatusColor(project.status)}>
                                    {project.status.replace('_', ' ')}
                                  </Badge>
                                  <Badge className={getPhaseColor(project.currentPhase)}>
                                    {project.currentPhase.replace('_', ' ')}
                                  </Badge>
                                </div>
                              </div>
                              <div className="mt-2">
                                <Progress value={project.progress || 0} className="h-2" />
                              </div>
                              <div className="text-xs text-muted-foreground mt-2">
                                Started: {formatTime(project.startTime)}
                              </div>
                            </CardContent>
                          </Card>
                        ))}
                      </div>
                    ) : (
                      <div className="text-center p-8 border rounded-lg border-dashed">
                        <p>No projects found. Create your first autonomous project!</p>
                      </div>
                    )}
                  </CardContent>
                  <CardFooter className="justify-between">
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={() => refetchProjects()}
                    >
                      <RefreshCw className="h-4 w-4 mr-2" />
                      Refresh
                    </Button>
                    {selectedProject && (
                      <div className="space-x-2">
                        <Button
                          variant="outline"
                          size="sm"
                          disabled={
                            projectControlMutation.isPending || 
                            projectDetailsLoading ||
                            (projectDetails?.status !== 'in_progress' && projectDetails?.status !== 'paused')
                          }
                          onClick={() => {
                            if (projectDetails?.status === 'in_progress') {
                              projectControlMutation.mutate({ 
                                id: selectedProject, 
                                action: 'pause' 
                              });
                            } else if (projectDetails?.status === 'paused') {
                              projectControlMutation.mutate({ 
                                id: selectedProject, 
                                action: 'resume' 
                              });
                            }
                          }}
                        >
                          {projectControlMutation.isPending ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : projectDetails?.status === 'in_progress' ? (
                            <Pause className="h-4 w-4" />
                          ) : (
                            <Play className="h-4 w-4" />
                          )}
                        </Button>
                        <Button
                          variant="destructive"
                          size="sm"
                          disabled={
                            projectControlMutation.isPending || 
                            projectDetailsLoading ||
                            projectDetails?.status === 'completed' || 
                            projectDetails?.status === 'failed'
                          }
                          onClick={() => {
                            projectControlMutation.mutate({ 
                              id: selectedProject, 
                              action: 'cancel' 
                            });
                          }}
                        >
                          Cancel
                        </Button>
                      </div>
                    )}
                  </CardFooter>
                </Card>
              </TabsContent>
              
              <TabsContent value="openmanus">
                <Card>
                  <CardHeader>
                    <CardTitle>OpenManus Agentic Projects</CardTitle>
                    <CardDescription>
                      Manage your advanced agentic system projects
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {openManusProjectsLoading ? (
                      <div className="flex items-center justify-center p-8">
                        <Loader2 className="h-8 w-8 animate-spin text-border" />
                      </div>
                    ) : openManusProjects && openManusProjects.length > 0 ? (
                      <div className="space-y-4">
                        {openManusProjects.map((project: any) => (
                          <Card 
                            key={project.project_id} 
                            className={`border-l-4 border-l-indigo-500 cursor-pointer hover:shadow-md transition-all`}
                          >
                            <CardContent className="p-4">
                              <div className="flex justify-between items-start">
                                <div>
                                  <h3 className="font-bold">{project.project_name}</h3>
                                  <p className="text-sm text-muted-foreground">{project.requirements}</p>
                                </div>
                                <div className="flex gap-2">
                                  <Badge className={getStatusColor(project.status)}>
                                    {project.status.replace('_', ' ')}
                                  </Badge>
                                </div>
                              </div>
                              <div className="mt-2">
                                <Progress value={project.progress || 0} className="h-2" />
                              </div>
                              <div className="text-xs text-muted-foreground mt-2">
                                Started: {project.started_at ? formatTime(project.started_at) : 'Not started yet'}
                              </div>
                              <div className="mt-2 flex flex-wrap gap-1">
                                <Badge variant="outline">Language: {project.language || 'Not specified'}</Badge>
                                <Badge variant="outline">Framework: {project.framework || 'Not specified'}</Badge>
                                <Badge variant="outline">Tasks: {project.subtask_counts?.total || 0}</Badge>
                                <Badge variant="outline" className="bg-green-100">Completed: {project.subtask_counts?.completed || 0}</Badge>
                                <Badge variant="outline" className="bg-yellow-100">In Progress: {project.subtask_counts?.in_progress || 0}</Badge>
                                <Badge variant="outline" className="bg-red-100">Failed: {project.subtask_counts?.failed || 0}</Badge>
                              </div>
                              <div className="mt-2">
                                <Button 
                                  variant="outline" 
                                  size="sm" 
                                  className="w-full"
                                  onClick={() => setSelectedOpenManusProject(project.project_id)}
                                >
                                  <Layers className="mr-2 h-4 w-4" />
                                  View Project Details
                                </Button>
                              </div>
                            </CardContent>
                          </Card>
                        ))}
                      </div>
                    ) : (
                      <div className="text-center p-8 border rounded-lg border-dashed">
                        <p>No OpenManus projects found. Create your first agentic project by selecting "OpenManus" in the "Primary Model" dropdown!</p>
                      </div>
                    )}
                  </CardContent>
                  <CardFooter>
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={() => refetchOpenManusProjects()}
                      className="w-full"
                    >
                      <RefreshCw className="h-4 w-4 mr-2" />
                      Refresh OpenManus Projects
                    </Button>
                  </CardFooter>
                </Card>
                
                {/* OpenManus Project Details */}
                {selectedOpenManusProject && (
                  <Card className="mt-4">
                    <CardHeader>
                      <CardTitle>OpenManus Project Details</CardTitle>
                      <CardDescription>
                        {openManusProjectDetailsLoading ? 'Loading...' : openManusProjectDetails?.project_name}
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      {openManusProjectDetailsLoading ? (
                        <div className="flex items-center justify-center p-8">
                          <Loader2 className="h-8 w-8 animate-spin text-border" />
                        </div>
                      ) : openManusProjectDetails ? (
                        <div className="space-y-4">
                          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <div>
                              <h3 className="font-semibold">Status</h3>
                              <Badge className={getStatusColor(openManusProjectDetails.status)}>
                                {openManusProjectDetails.status.replace('_', ' ')}
                              </Badge>
                            </div>
                            <div>
                              <h3 className="font-semibold">Progress</h3>
                              <div className="flex items-center gap-2">
                                <Progress value={openManusProjectDetails.progress || 0} className="h-2 flex-1" />
                                <span className="text-sm">{openManusProjectDetails.progress || 0}%</span>
                              </div>
                            </div>
                            <div>
                              <h3 className="font-semibold">Tasks</h3>
                              <div className="flex gap-2">
                                <Badge variant="outline" className="bg-green-100">
                                  {openManusProjectDetails.subtask_counts?.completed || 0} completed
                                </Badge>
                                <Badge variant="outline" className="bg-yellow-100">
                                  {openManusProjectDetails.subtask_counts?.in_progress || 0} in progress
                                </Badge>
                                <Badge variant="outline" className="bg-red-100">
                                  {openManusProjectDetails.subtask_counts?.failed || 0} failed
                                </Badge>
                              </div>
                            </div>
                          </div>
                          
                          <div>
                            <h3 className="font-semibold mb-2">Recent Logs</h3>
                            <div className="bg-muted rounded-md p-4 max-h-60 overflow-y-auto">
                              <pre className="text-sm">
                                {openManusProjectDetails.recent_logs?.map((log: any, index: number) => (
                                  <div key={index} className={`mb-1 ${log.level === 'error' ? 'text-red-500' : ''}`}>
                                    [{new Date(log.timestamp).toLocaleTimeString()}] {log.message}
                                  </div>
                                )).reverse() || 'No logs available'}
                              </pre>
                            </div>
                          </div>
                          
                          <div>
                            <Accordion type="single" collapsible className="w-full">
                              <AccordionItem value="files">
                                <AccordionTrigger>
                                  <div className="flex items-center">
                                    <FileText className="mr-2 h-4 w-4" />
                                    Generated Files
                                  </div>
                                </AccordionTrigger>
                                <AccordionContent>
                                  <div className="p-2 bg-muted rounded-md">
                                    {openManusProjectDetails.files ? 
                                      Object.keys(openManusProjectDetails.files).length > 0 ? (
                                        <div className="space-y-2">
                                          {Object.keys(openManusProjectDetails.files).map((filePath, index) => (
                                            <div key={index} className="flex items-center justify-between">
                                              <code className="text-sm">{filePath}</code>
                                              <Button variant="outline" size="sm">
                                                <Eye className="h-3 w-3 mr-1" />
                                                View
                                              </Button>
                                            </div>
                                          ))}
                                        </div>
                                      ) : (
                                        <p className="text-sm">No files generated yet.</p>
                                      )
                                    : <p className="text-sm">No files available.</p>}
                                  </div>
                                </AccordionContent>
                              </AccordionItem>
                            </Accordion>
                          </div>
                        </div>
                      ) : (
                        <div className="text-center p-4">
                          <p>No project details available</p>
                        </div>
                      )}
                    </CardContent>
                    <CardFooter>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => refetchOpenManusProjectDetails()}
                        className="w-full"
                      >
                        <RefreshCw className="h-4 w-4 mr-2" />
                        Refresh Details
                      </Button>
                    </CardFooter>
                  </Card>
                )}
              </TabsContent>
            </Tabs>
          </div>

          {/* Project Details */}
          {selectedProject && (
            <Card>
              <CardHeader>
                <CardTitle>Project Details</CardTitle>
                <CardDescription>
                  {projectDetailsLoading ? 'Loading...' : projectDetails?.name}
                </CardDescription>
              </CardHeader>
              <CardContent>
                {projectDetailsLoading ? (
                  <div className="flex items-center justify-center p-8">
                    <Loader2 className="h-8 w-8 animate-spin text-border" />
                  </div>
                ) : projectDetails ? (
                  <div className="space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div>
                        <h3 className="font-semibold">Status</h3>
                        <Badge className={getStatusColor(projectDetails.status)}>
                          {projectDetails.status.replace('_', ' ')}
                        </Badge>
                      </div>
                      <div>
                        <h3 className="font-semibold">Current Phase</h3>
                        <Badge className={getPhaseColor(projectDetails.currentPhase)}>
                          {projectDetails.currentPhase.replace('_', ' ')}
                        </Badge>
                      </div>
                      <div>
                        <h3 className="font-semibold">Progress</h3>
                        <div className="flex items-center gap-2">
                          <Progress value={projectDetails.progress || 0} className="h-2 flex-1" />
                          <span className="text-sm">{projectDetails.progress || 0}%</span>
                        </div>
                      </div>
                    </div>

                    <div>
                      <h3 className="font-semibold mb-2">Team Assignment</h3>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                        {Object.entries(projectDetails.teamAssignment || {}).map(([role, model]) => (
                          <Badge key={role} variant="outline" className="justify-between">
                            {role}: <span className="ml-1 font-normal">{model}</span>
                          </Badge>
                        ))}
                      </div>
                    </div>

                    <div>
                      <h3 className="font-semibold mb-2">Recent Logs</h3>
                      <div className="bg-muted rounded-md p-4 max-h-60 overflow-y-auto">
                        <pre className="text-sm">
                          {projectDetails.recentLogs?.map((log: any, index: number) => (
                            <div key={index} className={`mb-1 ${log.level === 'error' ? 'text-red-500' : ''}`}>
                              [{new Date(log.timestamp).toLocaleTimeString()}] [{log.phase.replace('_', ' ')}] {log.message}
                            </div>
                          )).reverse() || 'No logs available'}
                        </pre>
                      </div>
                    </div>

                    <div>
                      <h3 className="font-semibold mb-2">Artifacts</h3>
                      <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
                        <Button 
                          variant={projectDetails.artifactsSummary?.hasRequirements ? "default" : "outline"}
                          className="w-full"
                          disabled={!projectDetails.artifactsSummary?.hasRequirements}
                        >
                          Requirements
                        </Button>
                        <Button 
                          variant={projectDetails.artifactsSummary?.hasArchitecture ? "default" : "outline"}
                          className="w-full"
                          disabled={!projectDetails.artifactsSummary?.hasArchitecture}
                        >
                          Architecture
                        </Button>
                        <Button 
                          variant={projectDetails.artifactsSummary?.codeBaseSize > 0 ? "default" : "outline"}
                          className="w-full"
                          disabled={!projectDetails.artifactsSummary?.codeBaseSize}
                        >
                          Code ({projectDetails.artifactsSummary?.codeBaseSize || 0} files)
                        </Button>
                        <Button 
                          variant={projectDetails.artifactsSummary?.hasTestResults ? "default" : "outline"}
                          className="w-full"
                          disabled={!projectDetails.artifactsSummary?.hasTestResults}
                        >
                          Tests
                        </Button>
                        <Button 
                          variant={projectDetails.artifactsSummary?.hasDocumentation ? "default" : "outline"}
                          className="w-full"
                          disabled={!projectDetails.artifactsSummary?.hasDocumentation}
                        >
                          Documentation
                        </Button>
                      </div>
                    </div>

                    {projectDetails.qualityMetrics && (
                      <div>
                        <h3 className="font-semibold mb-2">Quality Metrics</h3>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                          {Object.entries(projectDetails.qualityMetrics).map(([key, value]) => (
                            <div key={key} className="bg-muted rounded-md p-3">
                              <div className="text-xs text-muted-foreground">{key.replace(/([A-Z])/g, ' $1').replace(/_/g, ' ').trim()}</div>
                              <div className="text-xl font-bold">{value}</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-center p-8">
                    <p>Project details not available</p>
                  </div>
                )}
              </CardContent>
              <CardFooter className="justify-end">
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={() => refetchProjectDetails()}
                >
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Refresh Details
                </Button>
              </CardFooter>
            </Card>
          )}
        </TabsContent>

        {/* Direct AI Coding Tab */}
        <TabsContent value="direct" className="space-y-8">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <Card>
              <CardHeader>
                <CardTitle>Generate Code</CardTitle>
                <CardDescription>
                  Generate code directly from requirements
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Form {...codeGenForm}>
                  <form onSubmit={codeGenForm.handleSubmit(onCodeGenSubmit)} className="space-y-4">
                    <FormField
                      control={codeGenForm.control}
                      name="requirements"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Requirements</FormLabel>
                          <FormControl>
                            <Textarea 
                              placeholder="Describe what code you want to generate" 
                              className="min-h-[150px]" 
                              {...field} 
                            />
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <FormField
                        control={codeGenForm.control}
                        name="language"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Language</FormLabel>
                            <FormControl>
                              <Input placeholder="e.g. JavaScript, Python" {...field} />
                            </FormControl>
                          </FormItem>
                        )}
                      />

                      <FormField
                        control={codeGenForm.control}
                        name="framework"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Framework</FormLabel>
                            <FormControl>
                              <Input placeholder="e.g. React, Django" {...field} />
                            </FormControl>
                          </FormItem>
                        )}
                      />
                    </div>

                    <FormField
                      control={codeGenForm.control}
                      name="primaryModel"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Model</FormLabel>
                          <Select 
                            onValueChange={field.onChange} 
                            defaultValue={field.value}
                          >
                            <FormControl>
                              <SelectTrigger>
                                <SelectValue placeholder="Select a model" />
                              </SelectTrigger>
                            </FormControl>
                            <SelectContent>
                              <SelectItem value="hybrid">Hybrid (Both Models)</SelectItem>
                              <SelectItem value="qwen2.5-7b-omni">Qwen2.5-7b-omni</SelectItem>
                              <SelectItem value="olympiccoder-7b">OlympicCoder-7B</SelectItem>
                            </SelectContent>
                          </Select>
                        </FormItem>
                      )}
                    />

                    <Button 
                      type="submit" 
                      className="w-full"
                      disabled={generateCodeMutation.isPending}
                    >
                      {generateCodeMutation.isPending ? (
                        <>
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          Generating Code...
                        </>
                      ) : (
                        'Generate Code'
                      )}
                    </Button>
                  </form>
                </Form>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Enhance Code</CardTitle>
                <CardDescription>
                  Optimize, refactor, document, test, or fix existing code
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Form {...enhanceForm}>
                  <form onSubmit={enhanceForm.handleSubmit(onEnhanceSubmit)} className="space-y-4">
                    <FormField
                      control={enhanceForm.control}
                      name="code"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Code</FormLabel>
                          <FormControl>
                            <Textarea 
                              placeholder="Paste your code here" 
                              className="min-h-[200px] font-mono" 
                              {...field} 
                            />
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <FormField
                        control={enhanceForm.control}
                        name="enhancement"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Enhancement Type</FormLabel>
                            <Select 
                              onValueChange={field.onChange} 
                              defaultValue={field.value}
                            >
                              <FormControl>
                                <SelectTrigger>
                                  <SelectValue placeholder="Select enhancement" />
                                </SelectTrigger>
                              </FormControl>
                              <SelectContent>
                                <SelectItem value="optimize">Optimize</SelectItem>
                                <SelectItem value="refactor">Refactor</SelectItem>
                                <SelectItem value="document">Document</SelectItem>
                                <SelectItem value="test">Add Tests</SelectItem>
                                <SelectItem value="fix">Fix Bugs</SelectItem>
                              </SelectContent>
                            </Select>
                          </FormItem>
                        )}
                      />

                      <FormField
                        control={enhanceForm.control}
                        name="language"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Language</FormLabel>
                            <FormControl>
                              <Input placeholder="e.g. JavaScript, Python" {...field} />
                            </FormControl>
                          </FormItem>
                        )}
                      />
                    </div>

                    <FormField
                      control={enhanceForm.control}
                      name="requirements"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Requirements (Optional)</FormLabel>
                          <FormControl>
                            <Textarea 
                              placeholder="Additional context or requirements" 
                              {...field} 
                            />
                          </FormControl>
                        </FormItem>
                      )}
                    />

                    <Button 
                      type="submit" 
                      className="w-full"
                      disabled={enhanceCodeMutation.isPending}
                    >
                      {enhanceCodeMutation.isPending ? (
                        <>
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          Enhancing Code...
                        </>
                      ) : (
                        `${enhanceForm.getValues('enhancement')}e Code`
                      )}
                    </Button>
                  </form>
                </Form>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* System Status Tab */}
        <TabsContent value="status" className="space-y-8">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <Card>
              <CardHeader>
                <CardTitle>Model Status</CardTitle>
                <CardDescription>
                  Status of the AI models
                </CardDescription>
              </CardHeader>
              <CardContent>
                {modelsLoading ? (
                  <div className="flex items-center justify-center p-8">
                    <Loader2 className="h-8 w-8 animate-spin text-border" />
                  </div>
                ) : modelStatus ? (
                  <div className="space-y-4">
                    {Object.entries(modelStatus).map(([modelName, modelData]: [string, any]) => (
                      <div key={modelName} className="border rounded-lg p-4">
                        <div className="flex justify-between items-center mb-2">
                          <h3 className="font-bold">{modelName}</h3>
                          <Badge 
                            className={
                              modelData.status === 'ready' 
                                ? 'bg-green-500' 
                                : modelData.status === 'busy' 
                                ? 'bg-yellow-500' 
                                : 'bg-red-500'
                            }
                          >
                            {modelData.status}
                          </Badge>
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-sm">
                          <div>
                            <span className="text-muted-foreground">Requests handled:</span>{' '}
                            {modelData.stats.requestsHandled}
                          </div>
                          <div>
                            <span className="text-muted-foreground">Failed requests:</span>{' '}
                            {modelData.stats.failedRequests}
                          </div>
                          <div>
                            <span className="text-muted-foreground">Avg response time:</span>{' '}
                            {Math.round(modelData.stats.averageResponseTime || 0)}ms
                          </div>
                          <div>
                            <span className="text-muted-foreground">Uptime:</span>{' '}
                            {Math.round(modelData.stats.uptimeSeconds / 60)} min
                          </div>
                        </div>
                        {modelData.stats.lastError && (
                          <div className="mt-2 text-sm text-red-500">
                            Last error: {modelData.stats.lastError}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center p-8 border rounded-lg border-dashed">
                    <p>Model status information not available</p>
                  </div>
                )}
              </CardContent>
              <CardFooter>
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={() => queryClient.invalidateQueries({ queryKey: ['/api/ai/models/status'] })}
                >
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Refresh Status
                </Button>
              </CardFooter>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>System Logs</CardTitle>
                <CardDescription>
                  Recent system activity
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="bg-muted rounded-md p-4 h-96 overflow-y-auto">
                  <div className="font-mono text-sm">
                    <div className="text-green-500">[System] Seren AI System initialized</div>
                    <div>[Qwen2.5-7b-omni] Model loaded successfully</div>
                    <div>[OlympicCoder-7B] Model loaded successfully</div>
                    <div className="text-yellow-500">[Warning] PyTorch acceleration not available, using CPU</div>
                    <div>[System] Continuous execution system ready</div>
                    <div>[System] Neurosymbolic reasoning system initialized</div>
                    <div>[System] Liquid neural network initialized</div>
                    <div>[System] Knowledge library initialized with baseline data</div>
                    <div>[System] Communication system ready for model collaboration</div>
                    <div className="text-blue-500">[Info] Dev team ready for autonomous software generation</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>About Seren AI</CardTitle>
              <CardDescription>
                The autonomous AI development system
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <p>
                Seren is a bleeding-edge autonomous AI development platform that combines 
                the strengths of multiple large language models to create a hyperintelligent 
                dev team capable of generating production-ready software from natural language 
                requirements.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div>
                  <h3 className="text-lg font-semibold mb-2">Core Components</h3>
                  <ul className="list-disc pl-5 space-y-1">
                    <li>Multi-model AI architecture (Qwen2.5-7b-omni + OlympicCoder-7B)</li>
                    <li>Neuro-symbolic reasoning framework</li>
                    <li>Liquid neural networks for adaptive intelligence</li>
                    <li>Metacognitive system for self-improvement</li>
                    <li>Knowledge library system for learning and sharing</li>
                    <li>Continuous execution pipeline</li>
                  </ul>
                </div>

                <div>
                  <h3 className="text-lg font-semibold mb-2">Features</h3>
                  <ul className="list-disc pl-5 space-y-1">
                    <li>Fully autonomous software development</li>
                    <li>Multiple collaboration modes between models</li>
                    <li>Sophisticated reasoning and problem-solving</li>
                    <li>Code generation, enhancement, and review</li>
                    <li>Architecture design and optimization</li>
                    <li>Comprehensive testing and documentation</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}