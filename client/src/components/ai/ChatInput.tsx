import { useState, useRef, FormEvent, KeyboardEvent } from "react";
import { Paperclip, Image, Mic, Play, FileCode, CornerDownLeft } from "lucide-react";
import { Button } from "@/components/ui/button";

interface ChatInputProps {
  onSendMessage: (content: string) => void;
  isLoading?: boolean;
}

export function ChatInput({ onSendMessage, isLoading = false }: ChatInputProps) {
  const [message, setMessage] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    
    if (message.trim() === "" || isLoading) return;
    
    onSendMessage(message);
    setMessage("");
    
    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Submit on Ctrl+Enter or Cmd+Enter
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleTextareaChange = () => {
    if (textareaRef.current) {
      // Reset height to auto to get the correct scrollHeight
      textareaRef.current.style.height = "auto";
      // Set height to scrollHeight to expand the textarea
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
      
      // Update state
      setMessage(textareaRef.current.value);
    }
  };

  return (
    <div className="border-t border-neutral-200 bg-white p-4 flex-shrink-0">
      <form onSubmit={handleSubmit} className="flex items-end">
        <div className="flex-1 bg-neutral-50 rounded-lg border border-neutral-200 p-3 overflow-hidden">
          <textarea
            ref={textareaRef}
            value={message}
            onChange={handleTextareaChange}
            onKeyDown={handleKeyDown}
            placeholder="Type your message or command here..."
            className="w-full bg-transparent resize-none outline-none text-neutral-800 text-sm max-h-32"
            rows={1}
            disabled={isLoading}
          />
          <div className="flex items-center justify-between mt-2 pt-2 border-t border-neutral-200">
            <div className="flex space-x-1">
              <button 
                type="button"
                className="p-1.5 text-neutral-500 hover:text-neutral-700 hover:bg-neutral-100 rounded-full"
                title="Attach file"
              >
                <Paperclip className="h-5 w-5" />
              </button>
              <button 
                type="button"
                className="p-1.5 text-neutral-500 hover:text-neutral-700 hover:bg-neutral-100 rounded-full"
                title="Upload image"
              >
                <Image className="h-5 w-5" />
              </button>
              <button 
                type="button"
                className="p-1.5 text-neutral-500 hover:text-neutral-700 hover:bg-neutral-100 rounded-full"
                title="Voice input"
              >
                <Mic className="h-5 w-5" />
              </button>
              <button 
                type="button"
                className="p-1.5 text-neutral-500 hover:text-neutral-700 hover:bg-neutral-100 rounded-full"
                title="Execute code"
              >
                <Play className="h-5 w-5" />
              </button>
              <button 
                type="button"
                className="p-1.5 text-neutral-500 hover:text-neutral-700 hover:bg-neutral-100 rounded-full"
                title="Insert code snippet"
              >
                <FileCode className="h-5 w-5" />
              </button>
            </div>
            <div className="text-xs text-neutral-500">Ctrl+Enter to submit</div>
          </div>
        </div>
        <Button 
          type="submit" 
          size="icon"
          className="ml-3 bg-primary-600 hover:bg-primary-700 text-white rounded-full p-3 shadow-sm flex-shrink-0 h-12 w-12"
          disabled={isLoading || message.trim() === ""}
        >
          {isLoading ? (
            <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
          ) : (
            <CornerDownLeft className="h-5 w-5" />
          )}
        </Button>
      </form>
    </div>
  );
}
