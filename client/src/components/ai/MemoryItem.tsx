import { AIMemory } from "@/lib/types";
import { format, formatDistanceToNow } from "date-fns";

interface MemoryItemProps {
  memory: AIMemory;
  onClick?: () => void;
}

export function MemoryItem({ memory, onClick }: MemoryItemProps) {
  const { title, content, timestamp } = memory;
  
  // Format the timestamp
  const timeDisplay = timestamp 
    ? formatDistanceToNow(new Date(timestamp), { addSuffix: true })
    : '';
  
  const fullTimeDisplay = timestamp
    ? format(new Date(timestamp), 'MMM d, yyyy h:mm a')
    : '';

  return (
    <div 
      className="memory-item bg-neutral-50 rounded-md p-3 transition-all duration-200 cursor-pointer"
      onClick={onClick}
      title={fullTimeDisplay}
    >
      <div className="flex justify-between items-start">
        <p className="text-xs font-medium text-neutral-700">{title}</p>
        <span className="text-xs text-neutral-500">{timeDisplay}</span>
      </div>
      <p className="text-xs text-neutral-600 mt-1 line-clamp-2">{content}</p>
    </div>
  );
}
