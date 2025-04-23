import { cn } from "@/lib/utils";

type ModelType = 'qwen' | 'olympic' | 'hybrid' | 'system';

interface ModelBadgeProps {
  model: ModelType;
  className?: string;
}

export function ModelBadge({ model, className }: ModelBadgeProps) {
  const baseClasses = "inline-flex items-center text-xs";
  const modelClasses = {
    qwen: "model-badge qwen",
    olympic: "model-badge olympic",
    hybrid: "model-badge hybrid",
    system: "model-badge system",
  };
  
  return (
    <span className={cn(baseClasses, modelClasses[model], className)}>
      {model === 'qwen' ? 'Qwen2.5-7b-omni' : 
       model === 'olympic' ? 'OlympicCoder-7B' : 
       model === 'hybrid' ? 'Hybrid' : 'System'}
    </span>
  );
}
