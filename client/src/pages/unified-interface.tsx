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
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const conversationId = useRef<string>('default-' + Date.now());
  
  // Project/task state
  const [projects, setProjects] = useState<Project[]>([]);
  const [backgroundTasks, setBackgroundTasks] = useState<BackgroundTask[]>([]);
  const [showProjectPanel, setShowProjectPanel] = useState(false);
  const [projectPrompt, setProjectPrompt] = useState('');
  
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

  // Handle WebSocket connection for real-time chat
  useEffect(() => {
    if (!user) return;
    
    let isMounted = true;
    
    // Initial connection - delay slightly to ensure user data is fully loaded
    setTimeout(() => {
      if (!isMounted) return; // Check if component is still mounted
      
      try {
        // Connect to WebSocket with user ID
        connectWebSocket(user.id);
        
        console.log("WebSocket connection established for user", user.id);
      } catch (error) {
        console.error("Error connecting to WebSocket:", error);
        
        toast({
          title: "Connection Issue",
          description: "Failed to establish real-time connection. Messages will be sent but responses may be delayed.",
          variant: "destructive",
        });
      }
    }, 500);
    
    // Register message handler
    const unregisterFn = registerMessageHandler((data) => {
      if (!isMounted) return;
      
      console.log("WebSocket message received:", data.type);
      
      if (data.type === 'new-message' && data.message) {
        // Add new message to chat
        setChatMessages(prev => [...prev, data.message as AIMessage]);
        
        // If it's an assistant response, we're no longer loading
        if (data.message.role === 'assistant') {
          setIsLoadingResponse(false);
        }
      }
    });
    
    // Cleanup function
    return () => {
      isMounted = false;
      
      if (unregisterFn) {
        unregisterFn();
      }
      
      try {
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
  
  // Toggle theme function
  const toggleTheme = () => {
    setTheme(theme === 'light' ? 'dark' : 'light');
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
  
  // Handle sending messages
  const sendMessage = async () => {
    if (!userInput.trim() || isLoadingResponse) return;
    
    // Create user message
    const userMessage: AIMessage = {
      conversationId: conversationId.current,
      role: 'user',
      content: userInput,
      model: selectedModel,
      userId: user?.id
    };
    
    // Add user message to state
    setChatMessages(prev => [...prev, userMessage]);
    
    // Clear input
    setUserInput('');
    
    // Show loading state
    setIsLoadingResponse(true);
    
    try {
      // Try WebSocket first
      let success = false;
      
      try {
        // Send via websocket
        success = sendChatMessage(userMessage);
        
        if (!success) {
          console.log("Message queued for delivery when connection is available");
        }
      } catch (wsError) {
        console.error("WebSocket error:", wsError);
        // We'll fall back to REST API call
      }
      
      // If WebSocket failed, use REST API as fallback
      if (!success) {
        try {
          // Fallback to REST API
          await apiRequest("POST", "/api/chat", {
            message: userMessage
          });
        } catch (apiError) {
          console.error("API error:", apiError);
          throw apiError;
        }
      }
    } catch (error) {
      console.error("Error sending message:", error);
      
      // Show error but don't stop loading state since we might still get a response
      toast({
        title: 'Connection Issue',
        description: 'Message sent but there may be a delay in receiving the response.',
        variant: 'default',
      });
      
      // Wait a bit and then stop loading if no response
      setTimeout(() => {
        setIsLoadingResponse(false);
      }, 5000);
    }
  };
  
  // Format messages with links and code blocks
  const formatMessage = (message: AIMessage) => {
    return message.content;
  };

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
            
            {/* Desktop Navigation */}
            <div className="hidden md:flex items-center space-x-4">
              <Button variant="ghost" className="focus-ring">
                <Terminal className="h-4 w-4 mr-2" />
                Dashboard
              </Button>
              <Button variant="ghost" className="focus-ring">
                <Code className="h-4 w-4 mr-2" />
                AI Workspace
              </Button>
              <Button variant="ghost" className="focus-ring">
                <FileText className="h-4 w-4 mr-2" />
                Documentation
              </Button>
            </div>
            
            {/* User Controls */}
            <div className="flex items-center space-x-2">
              {/* Theme Toggle */}
              <Button variant="outline" size="icon" onClick={toggleTheme} className="focus-ring">
                <Sun className="h-[1.2rem] w-[1.2rem] rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
                <Moon className="absolute h-[1.2rem] w-[1.2rem] rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
                <span className="sr-only">Toggle theme</span>
              </Button>
              
              {/* Model Selector */}
              <Select value={selectedModel} onValueChange={handleModelChange}>
                <SelectTrigger className="w-[150px]">
                  <SelectValue placeholder="Select Model" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="hybrid">Hybrid (Both Models)</SelectItem>
                  <SelectItem value="qwen">Qwen2.5-7b-omni</SelectItem>
                  <SelectItem value="olympic">OlympicCoder-7B</SelectItem>
                </SelectContent>
              </Select>
              
              {/* User Menu */}
              <Button variant="outline" className="rounded-full h-9 w-9 p-0">
                <span className="flex h-full w-full items-center justify-center rounded-full bg-primary/10 text-primary text-sm font-medium">
                  {user?.username?.charAt(0)?.toUpperCase() || 'U'}
                </span>
              </Button>
            </div>
          </div>
        </div>
      </nav>
      
      {/* Main Content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar - Desktop Only */}
        <div className="hidden lg:block w-64 border-r p-4">
          <div className="space-y-6">
            <div>
              <h3 className="text-sm font-medium mb-2">Navigation</h3>
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
              </div>
            </div>
          </div>
        </div>
        
        {/* Chat Container */}
        <div className="flex-1 flex flex-col">
          {/* Chat Messages */}
          <ScrollArea className="flex-1 p-4" ref={chatContainerRef}>
            <div className="space-y-6">
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
                    className={`max-w-[80%] rounded-lg px-4 py-3 shadow-sm ${
                      message.role === 'user' 
                        ? 'bg-primary text-primary-foreground' 
                        : 'glass border border-border/40'
                    }`}
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
                          {message.model === 'qwen' 
                            ? 'Qwen2.5-7b-omni' 
                            : message.model === 'olympic' 
                              ? 'OlympicCoder-7B' 
                              : 'Hybrid'}
                        </div>
                      </div>
                    )}
                  </div>
                  
                  {message.role === 'user' && (
                    <div className="flex h-8 w-8 ml-2 items-center justify-center rounded-full bg-primary text-primary-foreground">
                      <span className="text-xs font-medium">
                        {user?.username?.charAt(0)?.toUpperCase() || 'U'}
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
                  
                  <div className="max-w-[80%] rounded-lg px-4 py-3 glass border border-border/40">
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
          
          {/* Chat Input */}
          <div className="p-4 border-t">
            <form 
              className="flex gap-2 relative p-2 rounded-lg border focus-within:ring-1 focus-within:ring-primary"
              onSubmit={(e) => {
                e.preventDefault();
                sendMessage();
              }}
            >
              <Textarea
                placeholder="Send a message to Seren AI..."
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
          </div>
        </div>
      </div>
      
      {/* Footer */}
      <footer className="border-t py-4">
        <div className="container mx-auto px-4 text-center text-sm text-muted-foreground">
          <p>© {new Date().getFullYear()} Seren AI. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}