import { useMemo } from "react";
import { AIMessage } from "@/lib/types";
import { ModelBadge } from "./ModelBadge";
import { Copy, Cpu, User, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { format } from "date-fns";
import { cn } from "@/lib/utils";

interface ChatMessageProps {
  message: AIMessage;
  username: string;
}

export function ChatMessage({ message, username }: ChatMessageProps) {
  const { role, content, model, timestamp } = message;
  
  // Format timestamp
  const formattedTime = useMemo(() => {
    if (!timestamp) return '';
    
    const date = typeof timestamp === 'string' ? new Date(timestamp) : timestamp;
    return format(date, 'h:mm a');
  }, [timestamp]);
  
  // Get display name and CSS class based on role
  const messageConfig = useMemo(() => {
    switch (role) {
      case 'user':
        return {
          displayName: 'USER',
          cssClass: 'user-message',
          icon: <User className="h-5 w-5" />,
          bgColor: 'bg-primary-500'
        };
      case 'assistant':
        return {
          displayName: 'AI ASSISTANT',
          cssClass: 'ai-response',
          icon: <Cpu className="h-5 w-5" />,
          bgColor: 'bg-secondary-500'
        };
      case 'system':
        return {
          displayName: 'SYSTEM',
          cssClass: 'system-message',
          icon: <Zap className="h-5 w-5" />,
          bgColor: 'bg-accent-500'
        };
      default:
        return {
          displayName: 'UNKNOWN',
          cssClass: '',
          icon: <Cpu className="h-5 w-5" />,
          bgColor: 'bg-neutral-500'
        };
    }
  }, [role]);

  // Extract code blocks from content
  const formattedContent = useMemo(() => {
    if (role !== 'assistant' || !content.includes('```')) {
      return <p>{content}</p>;
    }
    
    const segments = content.split(/(```(?:[\w]*)\n[\s\S]*?\n```)/g);
    
    return (
      <>
        {segments.map((segment, index) => {
          if (segment.startsWith('```') && segment.endsWith('```')) {
            // Extract language and code
            const match = segment.match(/```([\w]*)\n([\s\S]*?)\n```/);
            if (!match) return <p key={index}>{segment}</p>;
            
            const [, language, code] = match;
            
            const isTerminal = (language === '' || language === 'sh' || language === 'bash') && 
                               (code.includes('$') || code.includes('#'));
            
            return (
              <div 
                key={index} 
                className={cn(
                  "mt-2 rounded-md p-3 font-mono text-xs overflow-x-auto",
                  isTerminal 
                    ? "bg-neutral-800 text-white" 
                    : "bg-neutral-50 text-neutral-800 border border-neutral-200"
                )}
              >
                <pre className={isTerminal ? "terminal-text" : ""}>{code}</pre>
              </div>
            );
          }
          
          return <p key={index} className="mb-2">{segment}</p>;
        })}
      </>
    );
  }, [content, role]);
  
  // Copy message to clipboard
  const copyToClipboard = () => {
    navigator.clipboard.writeText(content);
  };

  return (
    <div className={cn(messageConfig.cssClass, "bg-white rounded-lg p-4 mb-4 shadow-sm")}>
      <div className="flex items-start">
        {/* Avatar */}
        <div className={cn("flex-shrink-0 h-8 w-8 rounded-full flex items-center justify-center text-white", messageConfig.bgColor)}>
          {role === 'user' ? (
            <span className="font-semibold text-sm">{username.slice(0, 2).toUpperCase()}</span>
          ) : (
            messageConfig.icon
          )}
        </div>
        
        {/* Message content */}
        <div className="ml-3 flex-1">
          <p className="text-xs font-semibold text-neutral-500 mb-1">{messageConfig.displayName}</p>
          <div className="text-sm text-neutral-800">
            {formattedContent}
          </div>
          
          {/* Footer with model and timestamp */}
          <div className="mt-2 text-xs text-neutral-500 flex items-center">
            {model && role === 'assistant' && (
              <>
                <ModelBadge model={model as any} />
                <span className="mx-2">•</span>
              </>
            )}
            <span>{formattedTime}</span>
            
            {role === 'assistant' && (
              <Button variant="ghost" size="sm" className="ml-2 p-0 h-auto" onClick={copyToClipboard}>
                <Copy className="h-3.5 w-3.5 mr-1" />
                <span className="text-xs text-primary-600 hover:text-primary-800">Copy</span>
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
