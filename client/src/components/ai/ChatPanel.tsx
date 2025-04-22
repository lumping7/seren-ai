import { useState, useEffect, useRef } from "react";
import { ChatMessage } from "./ChatMessage";
import { ChatInput } from "./ChatInput";
import { AIMessage } from "@/lib/types";
import { Folder, Image, MoreVertical } from "lucide-react";
import { useAuth } from "@/hooks/use-auth";
import { connectWebSocket, sendChatMessage, registerMessageHandler } from "@/lib/websocket";
import { v4 as uuidv4 } from "uuid";

export function ChatPanel() {
  const { user } = useAuth();
  const [messages, setMessages] = useState<AIMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId] = useState(() => uuidv4());
  const chatMessagesRef = useRef<HTMLDivElement>(null);
  
  // Initialize with system message
  useEffect(() => {
    setMessages([
      {
        conversationId,
        role: "system",
        content: "NeurAI system initialized with Llama3 and Gemma3 models. Neuro-symbolic reasoning module active.\n\nWhat would you like to work on today?",
        model: "llama3",
        timestamp: new Date(),
      }
    ]);
  }, [conversationId]);
  
  // Connect to WebSocket and handle incoming messages
  useEffect(() => {
    if (!user) return;
    
    const socket = connectWebSocket(user.id);
    
    const unregister = registerMessageHandler((data) => {
      if (data.type === 'new-message' && data.message) {
        // Only add message if it belongs to the current conversation
        if (data.message.conversationId === conversationId) {
          setMessages(prev => [...prev, data.message]);
          setIsLoading(false);
        }
      }
    });
    
    return () => {
      unregister();
    };
  }, [user, conversationId]);
  
  // Scroll to bottom when messages change
  useEffect(() => {
    if (chatMessagesRef.current) {
      chatMessagesRef.current.scrollTop = chatMessagesRef.current.scrollHeight;
    }
  }, [messages]);
  
  const handleSendMessage = (content: string) => {
    if (!user) return;
    
    // Add user message to the chat
    const userMessage: AIMessage = {
      conversationId,
      role: "user",
      content,
      timestamp: new Date(),
    };
    
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    
    // Send message via WebSocket
    sendChatMessage({
      ...userMessage,
      userId: user.id
    });
  };

  return (
    <div className="w-full lg:w-2/3 flex flex-col overflow-hidden border-r border-neutral-200">
      {/* Chat Header */}
      <div className="bg-white h-12 border-b border-neutral-200 flex items-center px-4 flex-shrink-0">
        <h2 className="text-lg font-medium">AI Chat & Execution</h2>
        <div className="ml-auto flex items-center space-x-3">
          <button className="text-neutral-500 hover:text-neutral-700 p-1 rounded-full hover:bg-neutral-100">
            <Image className="h-5 w-5" />
          </button>
          <button className="text-neutral-500 hover:text-neutral-700 p-1 rounded-full hover:bg-neutral-100">
            <Folder className="h-5 w-5" />
          </button>
          <button className="text-neutral-500 hover:text-neutral-700 p-1 rounded-full hover:bg-neutral-100">
            <MoreVertical className="h-5 w-5" />
          </button>
        </div>
      </div>
      
      {/* Chat Messages */}
      <div 
        className="flex-1 overflow-y-auto bg-white p-4" 
        ref={chatMessagesRef}
      >
        {messages.map((message, index) => (
          <ChatMessage 
            key={index} 
            message={message} 
            username={user?.username || 'User'} 
          />
        ))}
        
        {/* Loading message */}
        {isLoading && (
          <div className="ai-response bg-white rounded-lg p-4 mb-4 shadow-sm">
            <div className="flex items-start">
              <div className="flex-shrink-0 h-8 w-8 rounded-full bg-secondary-500 flex items-center justify-center text-white">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
              <div className="ml-3 flex-1">
                <p className="text-xs font-semibold text-neutral-500 mb-1">AI ASSISTANT</p>
                <div className="text-sm text-neutral-800 flex items-center space-x-2">
                  <svg className="animate-spin h-5 w-5 text-neutral-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  <span>Processing...</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
      
      {/* Chat Input */}
      <ChatInput 
        onSendMessage={handleSendMessage}
        isLoading={isLoading}
      />
    </div>
  );
}
