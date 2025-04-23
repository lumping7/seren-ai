import React, { useState } from 'react';
import { apiRequest } from '@/lib/queryClient';
import { useToast } from '@/hooks/use-toast';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { Loader2, Send, BrainCircuit, Code, Zap } from 'lucide-react';
import Markdown from 'react-markdown';
import { cn } from '@/lib/utils';

type VirtualComputerResponse = {
  model: string;
  generated_text: string;
  metadata: {
    system: string;
    timestamp: string;
    model_version: string;
    operation_id?: string;
  };
};

export default function VirtualComputerPage() {
  const [prompt, setPrompt] = useState('');
  const [model, setModel] = useState('hybrid');
  const [loading, setLoading] = useState(false);
  const [conversation, setConversation] = useState<{
    role: 'user' | 'assistant';
    content: string;
    model?: string;
    timestamp?: string;
  }[]>([]);
  const { toast } = useToast();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!prompt.trim()) return;
    
    // Add user message to conversation
    setConversation((prev) => [
      ...prev,
      { role: 'user', content: prompt }
    ]);
    
    // Clear input
    setPrompt('');
    setLoading(true);
    
    try {
      const response = await apiRequest('POST', '/api/virtual-computer', {
        prompt,
        model,
        operationId: Date.now().toString() // simple operation ID
      });
      
      const data: VirtualComputerResponse = await response.json();
      
      // Add AI response to conversation
      setConversation((prev) => [
        ...prev,
        { 
          role: 'assistant', 
          content: data.generated_text,
          model: data.model,
          timestamp: data.metadata.timestamp 
        }
      ]);
    } catch (error) {
      console.error('Error communicating with virtual computer:', error);
      toast({
        title: 'Communication Error',
        description: 'Failed to communicate with the AI system. Please try again.',
        variant: 'destructive',
      });
    } finally {
      setLoading(false);
    }
  };

  const getModelIcon = (modelType: string) => {
    switch (modelType) {
      case 'qwen':
        return <BrainCircuit className="mr-2 h-4 w-4" />;
      case 'olympic':
        return <Code className="mr-2 h-4 w-4" />;
      case 'hybrid':
      default:
        return <Zap className="mr-2 h-4 w-4" />;
    }
  };

  return (
    <div className="container mx-auto py-8 px-4">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-2 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">Seren AI Virtual Computer</h1>
        <p className="text-gray-500 mb-8">
          Interact with the completely offline, self-contained AI system with multiple specialized models.
        </p>
        
        <div className="grid grid-cols-1 gap-8">
          <div className="bg-background/50 p-4 rounded-lg border shadow-sm">
            <div className="space-y-4 mb-4 max-h-[60vh] overflow-y-auto">
              {conversation.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  <BrainCircuit className="mx-auto h-12 w-12 mb-4 opacity-50" />
                  <p>Start a conversation with the virtual computer system.</p>
                  <p className="text-sm mt-2">Try asking about its capabilities or how it works!</p>
                </div>
              ) : (
                conversation.map((message, index) => (
                  <div 
                    key={index}
                    className={cn(
                      "p-4 rounded-lg",
                      message.role === 'user' 
                        ? "bg-primary/10 ml-8" 
                        : "bg-muted/50 mr-8 border"
                    )}
                  >
                    <div className="flex items-center mb-2">
                      <div className={cn(
                        "rounded-full w-8 h-8 flex items-center justify-center mr-2",
                        message.role === 'user' 
                          ? "bg-primary text-primary-foreground" 
                          : "bg-background border"
                      )}>
                        {message.role === 'user' ? (
                          <span className="text-sm font-semibold">You</span>
                        ) : (
                          getModelIcon(message.model || 'hybrid')
                        )}
                      </div>
                      <div>
                        <div className="font-semibold">
                          {message.role === 'user' ? 'You' : (
                            message.model === 'qwen' 
                              ? 'Qwen2.5-7b-omni' 
                              : message.model === 'olympic' 
                                ? 'OlympicCoder-7B' 
                                : 'Hybrid AI System'
                          )}
                        </div>
                        {message.timestamp && (
                          <div className="text-xs text-gray-500">
                            {new Date(message.timestamp).toLocaleTimeString()}
                          </div>
                        )}
                      </div>
                    </div>
                    <div className="ml-10">
                      {message.role === 'assistant' ? (
                        <div className="prose prose-sm dark:prose-invert max-w-none">
                          <Markdown>{message.content}</Markdown>
                        </div>
                      ) : (
                        <p>{message.content}</p>
                      )}
                    </div>
                  </div>
                ))
              )}
              {loading && (
                <div className="bg-muted/50 p-4 rounded-lg mr-8 border">
                  <div className="flex items-center mb-2">
                    <div className="bg-background border rounded-full w-8 h-8 flex items-center justify-center mr-2">
                      {getModelIcon(model)}
                    </div>
                    <div className="font-semibold">
                      {model === 'qwen' 
                        ? 'Qwen2.5-7b-omni' 
                        : model === 'olympic' 
                          ? 'OlympicCoder-7B' 
                          : 'Hybrid AI System'}
                    </div>
                  </div>
                  <div className="ml-10 flex items-center text-gray-500">
                    <Loader2 className="h-4 w-4 animate-spin mr-2" />
                    Thinking...
                  </div>
                </div>
              )}
            </div>
            
            <Card>
              <CardHeader className="pb-4">
                <CardTitle className="text-sm font-medium">Model Selection</CardTitle>
                <CardDescription>Choose which AI model to use</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-3 gap-2">
                  <Button 
                    variant={model === 'hybrid' ? 'default' : 'outline'} 
                    className="flex items-center justify-center"
                    onClick={() => setModel('hybrid')}
                  >
                    <Zap className="mr-2 h-4 w-4" />
                    Hybrid System
                  </Button>
                  <Button 
                    variant={model === 'qwen' ? 'default' : 'outline'} 
                    className="flex items-center justify-center"
                    onClick={() => setModel('qwen')}
                  >
                    <BrainCircuit className="mr-2 h-4 w-4" />
                    Qwen2.5-7b-omni
                  </Button>
                  <Button 
                    variant={model === 'olympic' ? 'default' : 'outline'} 
                    className="flex items-center justify-center"
                    onClick={() => setModel('olympic')}
                  >
                    <Code className="mr-2 h-4 w-4" />
                    OlympicCoder-7B
                  </Button>
                </div>
              </CardContent>
              <CardFooter>
                <form onSubmit={handleSubmit} className="flex w-full items-center space-x-2">
                  <Input
                    placeholder="Ask the virtual computer system..."
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    disabled={loading}
                    className="flex-1"
                  />
                  <Button type="submit" disabled={loading || !prompt.trim()}>
                    {loading ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Send className="h-4 w-4" />
                    )}
                  </Button>
                </form>
              </CardFooter>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}