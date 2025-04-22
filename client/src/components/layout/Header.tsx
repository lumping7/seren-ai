import { useState } from "react";
import { Bell, Moon, ChevronDown } from "lucide-react";
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger, 
  SelectValue
} from "@/components/ui/select";
import { useTheme } from "next-themes";
import { useLocation } from "wouter";

interface HeaderProps {
  username: string;
}

export function Header({ username }: HeaderProps) {
  const { setTheme } = useTheme();
  const [location] = useLocation();
  const [model, setModel] = useState<string>("hybrid");
  
  // Format the current path for breadcrumb
  const pathSegments = location.split('/').filter(Boolean);
  const pageName = pathSegments.length > 0 
    ? pathSegments[0].charAt(0).toUpperCase() + pathSegments[0].slice(1) 
    : "Dashboard";
  
  return (
    <header className="bg-white shadow-sm border-b border-neutral-200 h-16 flex-shrink-0">
      <div className="flex h-full px-4 items-center">
        {/* Breadcrumb */}
        <nav className="ml-4 hidden md:flex" aria-label="Breadcrumb">
          <ol className="flex items-center space-x-1 text-sm">
            <li>
              <a href="#" className="text-neutral-500 hover:text-primary-600">Dashboard</a>
            </li>
            {pathSegments.length > 0 && (
              <li className="flex items-center">
                <ChevronDown className="h-4 w-4 text-neutral-400 rotate-90" />
                <span className="ml-1 font-medium text-neutral-900">{pageName}</span>
              </li>
            )}
          </ol>
        </nav>
        
        {/* Model switcher */}
        <div className="ml-auto flex items-center space-x-4">
          <div className="flex items-center">
            <span className="mr-2 text-sm text-neutral-600 hidden sm:inline">AI Model:</span>
            <Select 
              value={model} 
              onValueChange={setModel}
            >
              <SelectTrigger className="w-40 h-9 text-sm bg-white">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="hybrid">Hybrid (Llama3 + Gemma3)</SelectItem>
                <SelectItem value="llama3">Llama3 Only</SelectItem>
                <SelectItem value="gemma3">Gemma3 Only</SelectItem>
              </SelectContent>
            </Select>
          </div>
          
          {/* Theme toggle */}
          <button 
            className="p-1.5 rounded-full text-neutral-500 hover:bg-neutral-100"
            onClick={() => setTheme('dark')}
          >
            <Moon className="h-5 w-5" />
          </button>
          
          {/* Notifications */}
          <button className="p-1.5 rounded-full text-neutral-500 hover:bg-neutral-100 relative">
            <Bell className="h-5 w-5" />
            <span className="absolute top-0 right-0 block h-2 w-2 rounded-full bg-red-500"></span>
          </button>
          
          {/* User avatar (mini) */}
          <div className="relative ml-2">
            <button className="flex items-center space-x-2">
              <div className="h-8 w-8 rounded-full bg-primary-500 flex items-center justify-center text-white font-semibold text-sm">
                {username.slice(0, 2).toUpperCase() || 'U'}
              </div>
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}
