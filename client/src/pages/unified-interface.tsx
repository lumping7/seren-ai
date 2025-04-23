import React, { useState, useEffect, useRef } from 'react';
import { useAuth } from '@/hooks/use-auth';
import { useToast } from '@/hooks/use-toast';
import { useTheme } from '@/components/providers/theme-provider';
import { useQuery, useMutation } from '@tanstack/react-query';
import { apiRequest, queryClient } from '@/lib/queryClient';
import { connectWebSocket, sendChatMessage, registerMessageHandler, closeWebSocket } from '@/lib/websocket';
import { AIMessage, Project, BackgroundTask } from '@/lib/types';

import { 
  Loader2, 
  Brain,
  Send,
  Download,
  Code,
  Terminal,
  FileText,
  Settings,
  Sun,
  Moon,
  Computer,
  ZapIcon,
  HelpCircle,
  LogOut,
  Menu,
  Home,
  Github,
  Mail,
  Twitter
} from 'lucide-react';

import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Textarea } from '@/components/ui/textarea';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover';
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from '@/components/ui/sheet';

export default function UnifiedInterface() {
  const { user, logoutMutation } = useAuth();
  const { toast } = useToast();
  const { theme, setTheme } = useTheme();
  
  // Chat state
  const [chatMessages, setChatMessages] = useState<AIMessage[]>([
    {
      conversationId: 'default',
      role: 'assistant',
      content: 'Hello! I\'m Seren AI, your advanced AI development platform. I can help you with coding, reasoning, problem-solving, and even create entire software projects from simple prompts. Just tell me what you need!'
    }
  ]);
  const [userInput, setUserInput] = useState('');
  const [selectedModel, setSelectedModel] = useState<'qwen' | 'olympic' | 'hybrid'>('hybrid');
  const [isLoadingResponse, setIsLoadingResponse] = useState(false);
  const [isCreatingProject, setIsCreatingProject] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const conversationId = useRef<string>('default-' + Date.now());
  
  // Project/task state
  const [projects, setProjects] = useState<Project[]>([]);
  const [backgroundTasks, setBackgroundTasks] = useState<BackgroundTask[]>([]);
  const [showProjectPanel, setShowProjectPanel] = useState(false);
  const [projectPrompt, setProjectPrompt] = useState('');
  
  // Handle WebSocket connection for real-time chat
  useEffect(() => {
    if (!user) return;
    
    let isMounted = true;
    let ws: WebSocket | null = null;
    let unregisterFn: (() => void) | null = null;
    let reconnectTimer: NodeJS.Timeout | null = null;
    
    const connectAndSetupWebSocket = () => {
      try {
        console.log("Setting up WebSocket connection...");
        
        // Clear any pending reconnect timers
        if (reconnectTimer) {
          clearTimeout(reconnectTimer);
          reconnectTimer = null;
        }
        
        // Close any existing WebSocket connection
        if (ws) {
          console.log("Closing existing WebSocket connection...");
          ws.onclose = null; // Prevent triggering reconnect
          ws.close();
          ws = null;
        }
        
        // Clean up any existing message handler
        if (unregisterFn) {
          unregisterFn();
          unregisterFn = null;
        }
        
        // Connect to WebSocket with user ID
        try {
          ws = connectWebSocket(user.id);
          
          // Set up event handlers only if ws is not null
          if (ws) {
            // Set up error handler
            const originalOnError = ws.onerror;
            ws.onerror = (event) => {
              if (originalOnError && ws) originalOnError.call(ws, event);
              
              if (!isMounted) return;
              
              console.error("WebSocket connection error", event);
              toast({
                title: "Connection Issue", 
                description: "Lost connection to server. Attempting to reconnect...",
                variant: "destructive",
              });
            };
            
            // Set up close handler
            const originalOnClose = ws.onclose;
            ws.onclose = (event) => {
              if (originalOnClose && ws) originalOnClose.call(ws, event);
              
              if (!isMounted) return;
              
              console.log(`WebSocket closed (${event.code}): ${event.reason || 'No reason given'}`);
              
              // Schedule reconnect
              if (!reconnectTimer && (event.code !== 1000 && event.code !== 1001)) {
                reconnectTimer = setTimeout(() => {
                  if (isMounted) {
                    console.log("Attempting to reconnect WebSocket...");
                    connectAndSetupWebSocket();
                  }
                }, 3000);
              }
            };
          } else {
            console.warn("Failed to create WebSocket connection");
          }
        } catch (wsError) {
          console.error("Error initializing WebSocket:", wsError);
        }
        
        // Register message handler
        unregisterFn = registerMessageHandler((data) => {
          if (!isMounted) return;
          
          console.log("WebSocket message received:", data.type);
          
          if (data.type === 'new-message' && data.message) {
            // Add new message to chat
            setChatMessages(prev => [...prev, data.message as AIMessage]);
            
            // If it's an assistant response, we're no longer loading
            if (data.message.role === 'assistant') {
              setIsLoadingResponse(false);
            }
          } else if (data.type === 'project-update' && data.projectId && data.update) {
            // Update projects list with new data
            setProjects(prev => 
              prev.map(p => p.id === data.projectId ? { ...p, ...data.update } : p)
            );
            
            // Show notification of update
            toast({
              title: 'Project Update',
              description: data.update.name && data.update.status ? 
                `${data.update.name}: ${data.update.status}` : 
                'Project updated',
            });
          } else if (data.type === 'task-complete') {
            // Remove completed task from background tasks
            setBackgroundTasks(prev => prev.filter(task => task.id !== data.taskId));
            
            // Show notification of completion
            toast({
              title: 'Task Complete',
              description: typeof data.message === 'string' ? 
                data.message : 
                'A background task has completed successfully.',
            });
          } else if (data.type === 'error') {
            // Show error notification
            toast({
              title: 'Error',
              description: data.error || 'An unknown error occurred',
              variant: 'destructive',
            });
          }
        });
      } catch (error) {
        console.error("Error setting up WebSocket:", error);
        
        if (isMounted) {
          toast({
            title: 'Connection Error',
            description: 'Failed to connect to the server. Will retry...',
            variant: 'destructive',
          });
          
          // Schedule reconnect
          if (!reconnectTimer) {
            reconnectTimer = setTimeout(() => {
              if (isMounted) {
                console.log("Attempting to reconnect after error...");
                connectAndSetupWebSocket();
              }
            }, 5000);
          }
        }
      }
    };
    
    // Initial connection
    connectAndSetupWebSocket();
    
    // Cleanup function
    return () => {
      console.log("Cleaning up WebSocket connections...");
      isMounted = false;
      
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
      }
      
      if (unregisterFn) {
        unregisterFn();
      }
      
      if (ws) {
        ws.onclose = null; // Prevent reconnect attempts
        ws.onerror = null;
        ws.close();
      }
      
      // Close any existing connection from websocket.ts
      try {
        // Use imported closeWebSocket function directly
        closeWebSocket();
      } catch (e) {
        console.error("Error calling closeWebSocket:", e);
      }
    };
  }, [user, toast]);
  
  // Scroll to bottom when new messages are added
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [chatMessages]);
  
  // Handle sending messages
  const sendMessage = async () => {
    if (!userInput.trim() || isLoadingResponse) return;
    
    const isProjectRequest = 
      userInput.toLowerCase().includes('create a') && 
      (userInput.toLowerCase().includes('project') || userInput.toLowerCase().includes('application') || 
       userInput.toLowerCase().includes('website') || userInput.toLowerCase().includes('app'));
    
    if (isProjectRequest && !isCreatingProject) {
      // Show project creation confirmation
      setProjectPrompt(userInput);
      setShowProjectPanel(true);
      return;
    }
    
    // Create user message
    const userMessage: AIMessage = {
      conversationId: conversationId.current,
      role: 'user',
      content: userInput,
      model: selectedModel,
      userId: user?.id,
      metadata: { 
        timestamp: new Date().toISOString(),
        isProjectRequest: isCreatingProject
      }
    };
    
    // Add user message to state
    setChatMessages(prev => [...prev, userMessage]);
    
    // Clear input
    setUserInput('');
    
    // Show loading state
    setIsLoadingResponse(true);
    
    try {
      // If creating a project, handle differently
      if (isCreatingProject) {
        setIsCreatingProject(false);
        
        // Create a new project
        const projectId = Date.now().toString();
        const newProject = {
          id: projectId,
          name: 'New Project',
          description: userInput,
          status: 'Creating',
          progress: 0,
          createdAt: new Date().toISOString(),
          lastUpdated: new Date().toISOString()
        };
        
        // Add to projects list
        setProjects(prev => [...prev, newProject]);
        
        // Add to background tasks
        setBackgroundTasks(prev => [
          ...prev, 
          { 
            id: projectId, 
            type: 'project-creation', 
            name: 'Project Creation', 
            status: 'In Progress',
            startTime: new Date().toISOString()
          }
        ]);
        
        // Create AI assistant message about project creation
        const assistantMessage: AIMessage = {
          conversationId: conversationId.current,
          role: 'assistant',
          content: `I'm now creating your project based on your description. You can continue chatting with me while I work on this in the background. You'll see updates in the Projects panel.`,
          model: selectedModel,
          metadata: { projectId }
        };
        
        // Add assistant message
        setChatMessages(prev => [...prev, assistantMessage]);
        
        // No longer loading
        setIsLoadingResponse(false);
        
        // Send websocket message to start the project creation
        try {
          sendChatMessage({
            type: 'project-create',
            message: userMessage,
            model: selectedModel,
            projectId
          });
        } catch (wsError) {
          console.error("WebSocket error when sending project message:", wsError);
          // Fall back to REST API
          sendMessageViaApi(userMessage, projectId);
        }
      } else {
        // Regular chat message
        
        // Try to send via websocket first
        try {
          sendChatMessage({
            type: 'chat-message',
            message: userMessage,
            model: selectedModel
          });
        } catch (wsError) {
          console.error("WebSocket error when sending chat message:", wsError);
          // Fall back to REST API
          sendMessageViaApi(userMessage);
        }
      }
    } catch (error) {
      console.error("Error sending message:", error);
      
      // Show error
      toast({
        title: 'Error',
        description: 'Failed to send message. Please try again.',
        variant: 'destructive',
      });
      
      setIsLoadingResponse(false);
    }
  };
  
  // Send message via API (fallback)
  const sendMessageViaApi = async (message: AIMessage, projectId?: string) => {
    try {
      const payload = {
        message,
        model: selectedModel,
        ...(projectId && { projectId })
      };
      
      const res = await apiRequest('POST', '/api/chat', payload);
      const data = await res.json();
      
      // Handle response
      if (data.message) {
        setChatMessages(prev => [...prev, data.message]);
      }
      
      // No longer loading
      setIsLoadingResponse(false);
    } catch (error) {
      console.error("API error:", error);
      
      // Show error
      toast({
        title: 'API Error',
        description: 'Failed to get response from server. Please try again.',
        variant: 'destructive',
      });
      
      setIsLoadingResponse(false);
    }
  };
  
  // Handle project creation
  const handleProjectCreation = () => {
    // Hide the panel
    setShowProjectPanel(false);
    
    // Set flag that we're creating a project
    setIsCreatingProject(true);
    
    // Send the message (with the existing prompt)
    sendMessage();
  };
  
  // Format messages with links and code blocks
  const formatMessage = (message: AIMessage) => {
    // Simple implementation: just return the content as-is for now
    // In a real implementation, you would parse for Markdown, code blocks, etc.
    return message.content;
  };
  
  // Handle model change
  const handleModelChange = (value: string) => {
    setSelectedModel(value as 'qwen' | 'olympic' | 'hybrid');
    toast({
      title: 'Model Changed',
      description: `Now using ${
        value === 'qwen' ? 'Qwen2.5-7b-omni' : 
        value === 'olympic' ? 'OlympicCoder-7B' : 
        'Hybrid (both models)'
      }`,
    });
  };
  
  // Toggle theme function
  const toggleTheme = () => {
    setTheme(theme === 'light' ? 'dark' : 'light');
  };
  
  // Mobile detection
  const [isMobile, setIsMobile] = useState(false);
  
  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 768);
    };
    
    checkMobile();
    
    // Add event listener for window resize
    window.addEventListener('resize', checkMobile);
    
    // Cleanup
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  return (
    <div className="flex flex-col h-screen bg-background">
      {/* Main Navigation */}
      <nav className="border-b bg-background/80 backdrop-blur-xl supports-[backdrop-filter]:bg-background/50 sticky top-0 z-50">
        <div className="container mx-auto px-4">
          <div className="flex items-center justify-between h-16">
            {/* Logo and Brand */}
            <div className="flex items-center gap-3">
              <div className="bg-primary/10 p-2 rounded-lg">
                <Brain className="h-5 w-5 sm:h-6 sm:w-6 text-primary" />
              </div>
              <div>
                <h1 className="text-lg sm:text-xl font-bold gradient-text">Seren AI</h1>
                <p className="text-xs text-muted-foreground hidden sm:block">Advanced AI Development System</p>
              </div>
            </div>
            
            {/* Primary Nav Links - Desktop */}
            <div className="hidden md:flex items-center space-x-4">
              <Button variant="ghost" className="focus-ring text-foreground hover:text-primary hover:bg-primary/10">
                <Terminal className="h-4 w-4 mr-2" />
                Dashboard
              </Button>
              <Button variant="ghost" className="focus-ring text-foreground hover:text-primary hover:bg-primary/10">
                <Code className="h-4 w-4 mr-2" />
                AI Workspace
              </Button>
              <Button variant="ghost" className="focus-ring text-foreground hover:text-primary hover:bg-primary/10">
                <FileText className="h-4 w-4 mr-2" />
                Documentation
              </Button>
              <Button variant="ghost" className="focus-ring text-foreground hover:text-primary hover:bg-primary/10">
                <HelpCircle className="h-4 w-4 mr-2" />
                Help
              </Button>
            </div>
            
            {/* Mobile Menu Trigger */}
            <div className="md:hidden flex items-center">
              <Sheet>
                <SheetTrigger asChild>
                  <Button variant="ghost" size="icon">
                    <Menu className="h-5 w-5" />
                  </Button>
                </SheetTrigger>
                <SheetContent side="left" className="w-[280px] glass">
                  <SheetHeader className="border-b pb-4 mb-4">
                    <SheetTitle className="flex items-center gap-2">
                      <Brain className="h-5 w-5 text-primary" />
                      <span className="gradient-text">Seren AI</span>
                    </SheetTitle>
                    <SheetDescription>
                      Advanced AI Development System
                    </SheetDescription>
                  </SheetHeader>
                  <div className="space-y-1">
                    <Button variant="ghost" className="w-full justify-start">
                      <Home className="h-4 w-4 mr-2" />
                      Home
                    </Button>
                    <Button variant="ghost" className="w-full justify-start">
                      <Terminal className="h-4 w-4 mr-2" />
                      Dashboard
                    </Button>
                    <Button variant="ghost" className="w-full justify-start">
                      <Code className="h-4 w-4 mr-2" />
                      AI Workspace
                    </Button>
                    <Button variant="ghost" className="w-full justify-start">
                      <FileText className="h-4 w-4 mr-2" />
                      Documentation
                    </Button>
                    <Button variant="ghost" className="w-full justify-start">
                      <HelpCircle className="h-4 w-4 mr-2" />
                      Help
                    </Button>
                  </div>
                  
                  <div className="absolute bottom-4 left-4 right-4">
                    <div className="space-y-2">
                      <div className="text-xs text-muted-foreground">
                        Current user: <span className="font-medium">{user.username}</span>
                      </div>
                      <Button 
                        variant="outline" 
                        size="sm" 
                        className="w-full justify-center gap-2"
                        onClick={() => logoutMutation.mutate()}
                      >
                        <LogOut className="h-4 w-4" />
                        Sign out
                      </Button>
                    </div>
                  </div>
                </SheetContent>
              </Sheet>
            </div>
            
            <div className="flex items-center space-x-1 sm:space-x-4">
              {/* Background Tasks Counter */}
              {backgroundTasks.length > 0 && (
                <Popover>
                  <PopoverTrigger asChild>
                    <Button variant="outline" size="sm" className="gap-1 sm:gap-2">
                      <ZapIcon className="h-4 w-4 text-yellow-500" />
                      <span className="hidden sm:inline">{backgroundTasks.length} Task{backgroundTasks.length > 1 ? 's' : ''}</span>
                      <span className="sm:hidden">{backgroundTasks.length}</span>
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-80 p-0">
                    <div className="p-4 border-b">
                      <h3 className="font-medium">Background Tasks</h3>
                    </div>
                    <ScrollArea className="h-[300px]">
                      <div className="p-4 space-y-3">
                        {backgroundTasks.map(task => (
                          <div key={task.id} className="flex items-center justify-between border rounded-lg p-3">
                            <div className="space-y-1">
                              <div className="font-medium">{task.name}</div>
                              <div className="text-sm text-muted-foreground">{task.status}</div>
                            </div>
                            <Badge variant={task.status === 'Running' ? 'default' : 'outline'}>
                              {task.status}
                            </Badge>
                          </div>
                        ))}
                      </div>
                    </ScrollArea>
                  </PopoverContent>
                </Popover>
              )}
              
              {/* Model Selector */}
              <Select value={selectedModel} onValueChange={handleModelChange}>
                <SelectTrigger className="w-[120px] sm:w-[180px] focus-ring">
                  <SelectValue placeholder={isMobile ? "Model" : "Select Model"} />
                </SelectTrigger>
                <SelectContent className="glass">
                  <SelectItem value="hybrid">
                    <div className="flex items-center gap-2">
                      <div className="h-2 w-2 rounded-full bg-purple-500"></div>
                      <span>{isMobile ? "Hybrid" : "Hybrid (Both Models)"}</span>
                    </div>
                  </SelectItem>
                  <SelectItem value="qwen">
                    <div className="flex items-center gap-2">
                      <div className="h-2 w-2 rounded-full bg-blue-500"></div>
                      <span>{isMobile ? "Qwen" : "Qwen2.5-7b-omni"}</span>
                    </div>
                  </SelectItem>
                  <SelectItem value="olympic">
                    <div className="flex items-center gap-2">
                      <div className="h-2 w-2 rounded-full bg-emerald-500"></div>
                      <span>{isMobile ? "Olympic" : "OlympicCoder-7B"}</span>
                    </div>
                  </SelectItem>
                </SelectContent>
              </Select>
              
              {/* Theme Toggle */}
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button variant="outline" size="icon" onClick={toggleTheme} className="focus-ring">
                      <Sun className="h-[1.2rem] w-[1.2rem] rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
                      <Moon className="absolute h-[1.2rem] w-[1.2rem] rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
                      <span className="sr-only">Toggle theme</span>
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent className="glass">
                    <p>Toggle theme</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              
              {/* Projects Button */}
              <Sheet>
                <SheetTrigger asChild>
                  <Button variant="outline" size="sm" className={`${isMobile ? "px-2" : ""} focus-ring`}>
                    {isMobile ? (
                      <FileText className="h-4 w-4" />
                    ) : (
                      <div className="flex items-center gap-2">
                        <FileText className="h-4 w-4" />
                        <span>Projects</span>
                      </div>
                    )}
                    {projects.length > 0 && (
                      <Badge variant="secondary" className={isMobile ? "ml-1" : "ml-2"}>{projects.length}</Badge>
                    )}
                  </Button>
                </SheetTrigger>
                <SheetContent side="right" className="w-[90vw] sm:w-[540px] glass border-l border-border/50">
                  <SheetHeader>
                    <SheetTitle className="text-xl gradient-text">Your Projects</SheetTitle>
                    <SheetDescription>
                      View and manage your AI-generated projects
                    </SheetDescription>
                  </SheetHeader>
                  <div className="mt-6 space-y-4">
                    {projects.length > 0 ? (
                      projects.map(project => (
                        <Card key={project.id} className={project.progress === 100 ? "border-green-500" : ""}>
                          <CardHeader className="pb-2">
                            <div className="flex justify-between items-start">
                              <div>
                                <CardTitle>{project.name}</CardTitle>
                                <CardDescription className="line-clamp-2">{project.description}</CardDescription>
                              </div>
                              <Badge variant={project.progress === 100 ? "outline" : "secondary"}>
                                {project.progress === 100 ? "Completed" : "In Progress"}
                              </Badge>
                            </div>
                          </CardHeader>
                          <CardContent className="pb-3">
                            <div className="space-y-2">
                              <div className="flex justify-between text-sm">
                                <span className="text-muted-foreground">Progress:</span>
                                <span>{project.progress}%</span>
                              </div>
                              <div className="h-2 bg-secondary rounded-full overflow-hidden">
                                <div 
                                  className="h-full bg-primary rounded-full"
                                  style={{ width: `${project.progress}%` }}
                                ></div>
                              </div>
                              <div className="flex justify-between text-sm">
                                <span className="text-muted-foreground">Status:</span>
                                <span>{project.status}</span>
                              </div>
                            </div>
                          </CardContent>
                          <CardFooter>
                            {project.progress === 100 && (
                              <Button className="w-full gap-2">
                                <Download className="h-4 w-4" />
                                Download Project
                              </Button>
                            )}
                          </CardFooter>
                        </Card>
                      ))
                    ) : (
                      <div className="text-center p-6 border rounded-lg">
                        <p className="text-muted-foreground">No projects yet. Ask Seren to create a project for you.</p>
                      </div>
                    )}
                  </div>
                </SheetContent>
              </Sheet>
              
              {/* User Menu */}
              <Popover>
                <PopoverTrigger asChild>
                  <Button variant="outline" className="focus-ring rounded-full h-9 w-9 p-0">
                    <span className="flex h-full w-full items-center justify-center rounded-full bg-primary/10 text-primary text-sm font-medium">
                      {user.username?.charAt(0)?.toUpperCase() || 'U'}
                    </span>
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="glass w-56 p-0" align="end">
                  <div className="p-3 border-b">
                    <div className="flex items-center gap-3">
                      <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10 text-primary text-sm font-medium">
                        {user.username?.charAt(0)?.toUpperCase() || 'U'}
                      </div>
                      <div>
                        <p className="text-sm font-medium">{user.username}</p>
                        <p className="text-xs text-muted-foreground">Admin User</p>
                      </div>
                    </div>
                  </div>
                  
                  <div className="p-2">
                    <Button 
                      variant="ghost" 
                      size="sm" 
                      className="w-full justify-start gap-2 focus-ring"
                      onClick={() => logoutMutation.mutate()}
                    >
                      <LogOut className="h-4 w-4" />
                      Sign out
                    </Button>
                  </div>
                </PopoverContent>
              </Popover>
            </div>
          </div>
        </div>
      </nav>
      
      {/* Main Content Area with Sidebar */}
      <div className="flex-1 overflow-hidden flex">
        {/* Sidebar - Hidden on mobile */}
        <div className="hidden lg:block w-64 border-r p-4 bg-background/60">
          <div className="space-y-6">
            <div>
              <h3 className="text-sm font-medium mb-2">Navigation</h3>
              <div className="space-y-1">
                <Button variant="ghost" className="w-full justify-start focus-ring text-foreground hover:text-primary hover:bg-primary/10">
                  <Home className="h-4 w-4 mr-2" />
                  Home
                </Button>
                <Button variant="ghost" className="w-full justify-start focus-ring text-foreground hover:text-primary hover:bg-primary/10">
                  <Terminal className="h-4 w-4 mr-2" />
                  Dashboard
                </Button>
                <Button variant="ghost" className="w-full justify-start focus-ring text-foreground hover:text-primary hover:bg-primary/10">
                  <Code className="h-4 w-4 mr-2" />
                  AI Workspace
                </Button>
              </div>
            </div>
            
            <div>
              <h3 className="text-sm font-medium mb-2">AI Models</h3>
              <div className="space-y-1">
                <div className="flex items-center gap-2 px-3 py-2 text-sm">
                  <div className="h-2 w-2 rounded-full bg-blue-500"></div>
                  <span>Qwen2.5-7b-omni</span>
                </div>
                <div className="flex items-center gap-2 px-3 py-2 text-sm">
                  <div className="h-2 w-2 rounded-full bg-emerald-500"></div>
                  <span>OlympicCoder-7B</span>
                </div>
                <div className="flex items-center gap-2 px-3 py-2 text-sm">
                  <div className="h-2 w-2 rounded-full bg-purple-500"></div>
                  <span>Hybrid Mode</span>
                </div>
              </div>
            </div>
            
            <div>
              <h3 className="text-sm font-medium mb-2">Projects</h3>
              {projects.length > 0 ? (
                <div className="space-y-2">
                  {projects.slice(0, 3).map(project => (
                    <div key={project.id} className="border rounded-md p-2">
                      <div className="font-medium text-sm truncate">{project.name}</div>
                      <div className="flex items-center justify-between mt-1">
                        <span className="text-xs text-muted-foreground">{project.progress}%</span>
                        <Badge variant="outline" className="text-xs">
                          {project.status}
                        </Badge>
                      </div>
                    </div>
                  ))}
                  {projects.length > 3 && (
                    <Button variant="ghost" className="w-full text-xs">
                      View all {projects.length} projects
                    </Button>
                  )}
                </div>
              ) : (
                <div className="text-sm text-muted-foreground">
                  No projects yet
                </div>
              )}
            </div>
          </div>
        </div>
        
        {/* Main Chat Container */}
        <div className="flex-1 overflow-hidden">
          <div className="container mx-auto h-full flex flex-col">
            {/* Chat Messages */}
            <ScrollArea 
              className="flex-1 p-2 sm:p-4" 
              ref={chatContainerRef}
            >
              <div className="space-y-4 sm:space-y-6">
                {chatMessages.map((message, index) => (
                  <div 
                    key={index}
                    className={`flex items-end ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    {message.role === 'assistant' && (
                      <div className="flex h-8 w-8 mr-2 items-center justify-center rounded-full bg-primary/10">
                        <Brain className="h-4 w-4 text-primary" />
                      </div>
                    )}
                    
                    <div 
                      className={`max-w-[90%] sm:max-w-[80%] rounded-lg px-3 py-2 sm:px-4 sm:py-3 shadow-sm ${
                        message.role === 'user' 
                          ? 'bg-primary text-primary-foreground' 
                          : 'glass border border-border/40'
                      } ${index === chatMessages.length - 1 ? 'fade-in' : ''}`}
                    >
                      {formatMessage(message)}
                      
                      {message.role === 'assistant' && message.model && (
                        <div className="flex items-center justify-end gap-1.5 mt-2">
                          <div className={`h-2 w-2 rounded-full ${
                            message.model === 'qwen' 
                              ? 'bg-blue-500' 
                              : message.model === 'olympic' 
                                ? 'bg-emerald-500' 
                                : 'bg-purple-500'
                          }`} />
                          <div className="text-xs text-muted-foreground">
                            {isMobile ? 
                              (message.model === 'qwen' 
                                ? 'Qwen' 
                                : message.model === 'olympic' 
                                  ? 'Olympic' 
                                  : 'Hybrid')
                              : 
                              (message.model === 'qwen' 
                                ? 'Qwen2.5-7b-omni' 
                                : message.model === 'olympic' 
                                  ? 'OlympicCoder-7B' 
                                  : 'Hybrid')
                            }
                          </div>
                        </div>
                      )}
                    </div>
                    
                    {message.role === 'user' && (
                      <div className="flex h-8 w-8 ml-2 items-center justify-center rounded-full bg-primary text-primary-foreground">
                        <span className="text-xs font-medium">
                          {user.username?.charAt(0)?.toUpperCase() || 'U'}
                        </span>
                      </div>
                    )}
                  </div>
                ))}
                
                {isLoadingResponse && (
                  <div className="flex justify-start items-end">
                    <div className="flex h-8 w-8 mr-2 items-center justify-center rounded-full bg-primary/10">
                      <Brain className="h-4 w-4 text-primary" />
                    </div>
                    
                    <div className="max-w-[90%] sm:max-w-[80%] rounded-lg px-3 py-2 sm:px-4 sm:py-3 glass border border-border/40 fade-in">
                      <div className="flex items-center gap-2">
                        <div className="flex space-x-1">
                          <div className="h-2 w-2 rounded-full bg-primary animate-pulse"></div>
                          <div className="h-2 w-2 rounded-full bg-primary animate-pulse delay-150"></div>
                          <div className="h-2 w-2 rounded-full bg-primary animate-pulse delay-300"></div>
                        </div>
                        <span className="text-sm">Seren is thinking...</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </ScrollArea>
            
            {/* Project Creation Panel */}
            {showProjectPanel && (
              <div className="border-t p-3 sm:p-4 bg-card">
                <Card>
                  <CardHeader className="pb-2 px-3 sm:px-6">
                    <CardTitle className="text-lg sm:text-xl">Create New Project</CardTitle>
                    <CardDescription>
                      I'll generate a complete software project based on your requirements
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="px-3 sm:px-6">
                    <div className="space-y-2">
                      <p className="text-sm font-medium">Your project description:</p>
                      <div className="p-2 sm:p-3 bg-muted rounded-md text-sm sm:text-base">
                        {projectPrompt}
                      </div>
                    </div>
                  </CardContent>
                  <CardFooter className="flex justify-end gap-2 px-3 sm:px-6 pt-2">
                    <Button variant="outline" size={isMobile ? "sm" : "default"} onClick={() => setShowProjectPanel(false)}>
                      Cancel
                    </Button>
                    <Button size={isMobile ? "sm" : "default"} onClick={handleProjectCreation}>
                      Create Project
                    </Button>
                  </CardFooter>
                </Card>
              </div>
            )}
            
            {/* Chat Input */}
            <div className="p-3 sm:p-4 border-t bg-muted/20">
              <form 
                className="flex gap-2 relative glass p-1 sm:p-2 rounded-lg overflow-hidden focus-within:ring-1 focus-within:ring-primary/50"
                onSubmit={(e) => {
                  e.preventDefault();
                  sendMessage();
                }}
              >
                <Textarea
                  placeholder={isMobile ? "Message Seren AI..." : "Send a message to Seren AI... (Try 'Create a TODO list app with React')"}
                  value={userInput}
                  onChange={(e) => setUserInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      sendMessage();
                    }
                  }}
                  className="min-h-[50px] sm:min-h-[60px] flex-1 resize-none text-sm sm:text-base border-0 focus-visible:ring-0 bg-transparent px-3 py-2"
                  disabled={isLoadingResponse}
                />
                <div className="absolute right-2 bottom-2 sm:right-3 sm:bottom-3">
                  <Button 
                    type="submit"
                    size="icon" 
                    variant={userInput.trim() ? "default" : "ghost"}
                    className="h-9 w-9 rounded-full" 
                    disabled={isLoadingResponse || !userInput.trim()}
                  >
                    {isLoadingResponse ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Send className="h-4 w-4" />
                    )}
                  </Button>
                </div>
              </form>
              <div className="mt-3 text-xs text-center text-muted-foreground">
                <p className="line-clamp-2 sm:line-clamp-none">
                  <span className="opacity-80">Seren is designed to assist with software development, answer questions, and create projects.</span>
                  {!isMobile && <span className="opacity-70"> Ask me to create software, answer questions, or explain concepts!</span>}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Footer */}
      <footer className="border-t py-4 bg-background/70 backdrop-blur-md">
        <div className="container mx-auto px-4">
          <div className="flex flex-col md:flex-row justify-between items-center gap-4">
            <div className="flex items-center gap-2">
              <Brain className="h-5 w-5 text-primary" />
              <span className="font-medium">Seren AI</span>
            </div>
            
            <div className="flex items-center gap-6">
              <a href="#" className="text-sm text-muted-foreground hover:text-foreground">About</a>
              <a href="#" className="text-sm text-muted-foreground hover:text-foreground">Documentation</a>
              <a href="#" className="text-sm text-muted-foreground hover:text-foreground">Privacy</a>
              <a href="#" className="text-sm text-muted-foreground hover:text-foreground">Terms</a>
            </div>
            
            <div className="flex items-center gap-4">
              <a href="#" className="text-muted-foreground hover:text-foreground">
                <Github className="h-4 w-4" />
                <span className="sr-only">GitHub</span>
              </a>
              <a href="#" className="text-muted-foreground hover:text-foreground">
                <Twitter className="h-4 w-4" />
                <span className="sr-only">Twitter</span>
              </a>
              <a href="#" className="text-muted-foreground hover:text-foreground">
                <Mail className="h-4 w-4" />
                <span className="sr-only">Email</span>
              </a>
            </div>
          </div>
          
          <div className="mt-4 pt-4 border-t text-center text-xs text-muted-foreground">
            <p>© {new Date().getFullYear()} Seren AI. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}