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
  HelpCircle
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
        
        // Add to chat
        setChatMessages(prev => [...prev, assistantMessage]);
        setIsLoadingResponse(false);
        
        // Simulate project creation (in real version, this would be a backend API call)
        simulateProjectCreation(projectId, userInput);
        
        return;
      }
      
      // Try to send through WebSocket first
      let success = false;
      try {
        success = sendChatMessage(userMessage);
        console.log("WebSocket message send attempt result:", success);
      } catch (wsError) {
        console.error("WebSocket send error:", wsError);
        success = false;
      }
      
      // If WebSocket fails or isn't connected, fall back to REST API
      if (!success) {
        console.log("Falling back to REST API for message");
        try {
          const response = await apiRequest('POST', '/api/messages', userMessage);
          
          if (!response.ok) {
            throw new Error(`API error: ${response.status} ${response.statusText}`);
          }
          
          const assistantMessage = await response.json();
          
          // Add assistant response to chat
          setChatMessages(prev => [...prev, assistantMessage]);
          setIsLoadingResponse(false);
        } catch (apiError) {
          console.error("API fallback error:", apiError);
          
          // Create a synthetic message to show error to user
          const errorMessage: AIMessage = {
            conversationId: conversationId.current,
            role: 'assistant',
            content: "I apologize, but I'm having trouble connecting to the AI models right now. Please try again in a moment.",
            model: selectedModel,
            metadata: { error: true, timestamp: new Date().toISOString() }
          };
          
          setChatMessages(prev => [...prev, errorMessage]);
          setIsLoadingResponse(false);
          
          toast({
            title: 'Connection Issue',
            description: 'Could not reach the AI service. Please try again later.',
            variant: 'destructive',
          });
        }
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
  
  // Simulate project creation (temporary for demonstration)
  const simulateProjectCreation = (projectId: string, prompt: string) => {
    const steps = [
      { status: 'Analyzing requirements', progress: 10 },
      { status: 'Designing architecture', progress: 25 },
      { status: 'Generating code', progress: 40 },
      { status: 'Implementing features', progress: 65 },
      { status: 'Testing functionality', progress: 80 },
      { status: 'Finalizing project', progress: 95 },
      { status: 'Completed', progress: 100 }
    ];
    
    // Generate project name from prompt
    const generateProjectName = (prompt: string) => {
      const keywords = prompt.toLowerCase().split(' ');
      let name = '';
      
      if (prompt.toLowerCase().includes('website')) {
        name = 'Website';
      } else if (prompt.toLowerCase().includes('app')) {
        name = 'App';
      } else if (prompt.toLowerCase().includes('dashboard')) {
        name = 'Dashboard';
      } else if (prompt.toLowerCase().includes('api')) {
        name = 'API';
      } else {
        name = 'Project';
      }
      
      // Add a relevant descriptor
      for (const word of ['ecommerce', 'blog', 'social', 'chat', 'data', 'finance', 'health', 'task']) {
        if (keywords.includes(word)) {
          name = word.charAt(0).toUpperCase() + word.slice(1) + ' ' + name;
          break;
        }
      }
      
      return name;
    };
    
    const projectName = generateProjectName(prompt);
    
    // Update project name immediately
    setProjects(prev => 
      prev.map(p => p.id === projectId ? { ...p, name: projectName } : p)
    );
    
    // Simulate steps with delays
    steps.forEach((step, index) => {
      setTimeout(() => {
        setProjects(prev => 
          prev.map(p => 
            p.id === projectId 
              ? { 
                  ...p, 
                  status: step.status, 
                  progress: step.progress,
                  lastUpdated: new Date().toISOString()
                } 
              : p
          )
        );
        
        // If completed, remove from background tasks
        if (index === steps.length - 1) {
          setBackgroundTasks(prev => prev.filter(task => task.id !== projectId));
          
          // Add completion message to chat
          const completionMessage: AIMessage = {
            conversationId: conversationId.current,
            role: 'assistant',
            content: `I've completed your project "${projectName}" based on your requirements. You can now download it or view the details in the Projects panel.`,
            model: selectedModel,
            metadata: { projectId, completed: true }
          };
          
          setChatMessages(prev => [...prev, completionMessage]);
        }
      }, 3000 * (index + 1)); // Simulate each step taking 3 seconds
    });
  };
  
  // Handle project creation confirmation
  const handleProjectCreation = () => {
    setUserInput(projectPrompt);
    setIsCreatingProject(true);
    setShowProjectPanel(false);
    // Call sendMessage after state updates
    setTimeout(() => sendMessage(), 0);
  };
  
  // Handle executing code directly through chat
  const executeCodeFromChat = (code: string) => {
    const taskId = Date.now().toString();
    
    // Add to background tasks
    setBackgroundTasks(prev => [
      ...prev, 
      { 
        id: taskId, 
        type: 'code-execution', 
        name: 'Code Execution', 
        status: 'Running',
        startTime: new Date().toISOString()
      }
    ]);
    
    // In real implementation, this would call a backend API
    // Simulate execution with a timeout
    setTimeout(() => {
      const result = `// Execution complete\n// Output:\nExecuted ${code.length} bytes of code successfully`;
      
      // Remove from background tasks
      setBackgroundTasks(prev => prev.filter(task => task.id !== taskId));
      
      // Add result message to chat
      const resultMessage: AIMessage = {
        conversationId: conversationId.current,
        role: 'assistant',
        content: `\`\`\`\n${result}\n\`\`\``,
        model: selectedModel,
        metadata: { taskId, execution: true }
      };
      
      setChatMessages(prev => [...prev, resultMessage]);
    }, 2000);
  };
  
  // Theme toggle
  const toggleTheme = () => {
    setTheme(theme === 'dark' ? 'light' : 'dark');
  };
  
  // Handle model change
  const handleModelChange = (value: string) => {
    setSelectedModel(value as 'qwen' | 'olympic' | 'hybrid');
    toast({
      title: 'Model Changed',
      description: `Now using ${value === 'hybrid' ? 'Hybrid (Both Models)' : value === 'qwen' ? 'Qwen2.5-7b-omni' : 'OlympicCoder-7B'}`,
    });
  };
  
  // Format chat message display
  const formatMessage = (message: AIMessage) => {
    // Handle code blocks with syntax highlighting
    let content = message.content;
    const codeBlockRegex = /\`\`\`([\s\S]*?)\`\`\`/g;
    
    if (codeBlockRegex.test(content)) {
      content = content.replace(codeBlockRegex, (match, code) => {
        return `<div class="bg-muted p-4 rounded my-2 overflow-auto"><pre><code>${code}</code></pre></div>`;
      });
    }
    
    // Handle inline code
    const inlineCodeRegex = /\`([^\`]+)\`/g;
    if (inlineCodeRegex.test(content)) {
      content = content.replace(inlineCodeRegex, (match, code) => {
        return `<span class="bg-muted px-1 rounded font-mono text-sm">${code}</span>`;
      });
    }
    
    return (
      <div 
        className="prose dark:prose-invert max-w-none" 
        dangerouslySetInnerHTML={{ __html: content }}
      />
    );
  };
  
  // Loading state
  if (!user) {
    return (
      <div className="h-screen flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }
  
  // Function to determine if the viewport is mobile
  const [isMobile, setIsMobile] = useState<boolean>(false);
  
  // Set up responsive detection
  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 768);
    };
    
    // Initial check
    checkMobile();
    
    // Add event listener for window resize
    window.addEventListener('resize', checkMobile);
    
    // Cleanup
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  return (
    <div className="flex flex-col h-screen bg-background">
      {/* Header */}
      <header className="border-b px-2 sm:px-4 py-2 sm:py-3">
        <div className="container mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Brain className="h-6 w-6 sm:h-8 sm:w-8 text-primary" />
            <div>
              <h1 className="text-lg sm:text-xl font-bold">Seren AI</h1>
              <p className="text-xs text-muted-foreground hidden sm:block">Advanced AI Development System</p>
            </div>
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
              <SelectTrigger className="w-[120px] sm:w-[180px]">
                <SelectValue placeholder={isMobile ? "Model" : "Select Model"} />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="hybrid">Hybrid (Both Models)</SelectItem>
                <SelectItem value="qwen">Qwen2.5-7b-omni</SelectItem>
                <SelectItem value="olympic">OlympicCoder-7B</SelectItem>
              </SelectContent>
            </Select>
            
            {/* Theme Toggle */}
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="outline" size="icon" onClick={toggleTheme}>
                    {theme === 'dark' ? (
                      <Sun className="h-5 w-5" />
                    ) : theme === 'light' ? (
                      <Moon className="h-5 w-5" />
                    ) : (
                      <Computer className="h-5 w-5" />
                    )}
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Toggle theme: {theme}</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
            
            {/* Projects Button */}
            <Sheet>
              <SheetTrigger asChild>
                <Button variant="outline" size="sm" className={isMobile ? "px-2" : ""}>
                  {isMobile ? (
                    <FileText className="h-4 w-4" />
                  ) : (
                    <>Projects</>
                  )}
                  {projects.length > 0 && (
                    <Badge variant="secondary" className={isMobile ? "ml-1" : "ml-2"}>{projects.length}</Badge>
                  )}
                </Button>
              </SheetTrigger>
              <SheetContent side="right" className="w-[90vw] sm:w-[540px]">
                <SheetHeader>
                  <SheetTitle>Your Projects</SheetTitle>
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
            
            {/* User Info & Logout */}
            <div className="flex items-center gap-2">
              <div className="text-right">
                <p className="text-sm font-medium">{user.username}</p>
                <Button 
                  variant="link" 
                  size="sm" 
                  className="p-0 h-auto text-xs text-muted-foreground"
                  onClick={() => logoutMutation.mutate()}
                >
                  Sign out
                </Button>
              </div>
            </div>
          </div>
        </div>
      </header>
      
      {/* Main Chat Container */}
      <div className="flex-1 overflow-hidden">
        <div className="container mx-auto h-full flex flex-col">
          {/* Chat Messages */}
          <ScrollArea 
            className="flex-1 p-4" 
            ref={chatContainerRef}
          >
            <div className="space-y-6">
              {chatMessages.map((message, index) => (
                <div 
                  key={index}
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div 
                    className={`max-w-[80%] rounded-lg px-4 py-3 ${
                      message.role === 'user' 
                        ? 'bg-primary text-primary-foreground' 
                        : 'bg-muted'
                    }`}
                  >
                    {formatMessage(message)}
                    
                    {message.role === 'assistant' && message.model && (
                      <div className="mt-2 text-xs text-right text-muted-foreground">
                        {message.model === 'qwen' 
                          ? 'Qwen2.5-7b-omni' 
                          : message.model === 'olympic' 
                            ? 'OlympicCoder-7B' 
                            : 'Hybrid'}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              
              {isLoadingResponse && (
                <div className="flex justify-start">
                  <div className="max-w-[80%] rounded-lg px-4 py-3 bg-muted">
                    <div className="flex items-center gap-2">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      <span>Seren is thinking...</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </ScrollArea>
          
          {/* Project Creation Panel */}
          {showProjectPanel && (
            <div className="border-t p-4 bg-card">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle>Create New Project</CardTitle>
                  <CardDescription>
                    I'll generate a complete software project based on your requirements
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <p className="text-sm font-medium">Your project description:</p>
                    <div className="p-3 bg-muted rounded-md">
                      {projectPrompt}
                    </div>
                  </div>
                </CardContent>
                <CardFooter className="flex justify-end gap-2">
                  <Button variant="outline" onClick={() => setShowProjectPanel(false)}>
                    Cancel
                  </Button>
                  <Button onClick={handleProjectCreation}>
                    Create Project
                  </Button>
                </CardFooter>
              </Card>
            </div>
          )}
          
          {/* Chat Input */}
          <div className="p-4 border-t">
            <form 
              className="flex gap-2"
              onSubmit={(e) => {
                e.preventDefault();
                sendMessage();
              }}
            >
              <Textarea
                placeholder="Send a message to Seren AI... (Try 'Create a TODO list app with React')"
                value={userInput}
                onChange={(e) => setUserInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                  }
                }}
                className="min-h-[60px] flex-1 resize-none"
                disabled={isLoadingResponse}
              />
              <Button 
                type="submit"
                size="icon" 
                className="h-[60px] w-[60px]" 
                disabled={isLoadingResponse || !userInput.trim()}
              >
                {isLoadingResponse ? (
                  <Loader2 className="h-6 w-6 animate-spin" />
                ) : (
                  <Send className="h-6 w-6" />
                )}
              </Button>
            </form>
            <div className="mt-2 text-xs text-center text-muted-foreground">
              <p>
                Seren is designed to assist with software development, answer questions, and create projects.
                Ask me to create software, answer questions, or explain concepts!
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}