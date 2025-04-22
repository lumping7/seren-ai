import { cn } from "@/lib/utils";

type ModelType = 'llama3' | 'gemma3' | 'hybrid' | 'system';

interface ModelBadgeProps {
  model: ModelType;
  className?: string;
}

export function ModelBadge({ model, className }: ModelBadgeProps) {
  const baseClasses = "inline-flex items-center text-xs";
  const modelClasses = {
    llama3: "model-badge llama",
    gemma3: "model-badge gemma",
    hybrid: "model-badge hybrid",
    system: "model-badge system",
  };
  
  return (
    <span className={cn(baseClasses, modelClasses[model], className)}>
      {model === 'llama3' ? 'Llama3' : 
       model === 'gemma3' ? 'Gemma3' : 
       model === 'hybrid' ? 'Hybrid' : 'System'}
    </span>
  );
}
