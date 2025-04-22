import { useState } from "react";
import { Link, useLocation } from "wouter";
import { useAuth } from "@/hooks/use-auth";
import { cn } from "@/lib/utils";
import {
  Brain,
  Code2,
  Cpu,
  FileStack,
  Layers,
  Lock,
  LucideIcon,
  RefreshCw,
  Settings,
  Shield,
  Timer,
  Zap
} from "lucide-react";

type NavItemProps = {
  href: string;
  icon: LucideIcon;
  label: string;
  isActive: boolean;
};

function NavItem({ href, icon: Icon, label, isActive }: NavItemProps) {
  return (
    <li>
      <Link href={href}>
        <a
          className={cn(
            "flex items-center px-3 py-2 rounded-md mb-1",
            isActive 
              ? "bg-primary-50 text-primary-800" 
              : "text-neutral-700 hover:bg-neutral-100"
          )}
        >
          <Icon className="h-5 w-5 mr-3" />
          {label}
        </a>
      </Link>
    </li>
  );
}

export function Sidebar() {
  const [location] = useLocation();
  const { user, logoutMutation } = useAuth();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  
  const handleLogout = () => {
    logoutMutation.mutate();
  };
  
  const sections = [
    {
      title: "AI Core",
      items: [
        { href: "/", icon: Cpu, label: "Dashboard" },
        { href: "/memory", icon: Brain, label: "Memory System" },
        { href: "/reasoning", icon: FileStack, label: "Reasoning Engine" },
        { href: "/models", icon: Layers, label: "AI Models" },
      ]
    },
    {
      title: "Evolution",
      items: [
        { href: "/upgrade", icon: RefreshCw, label: "Self-Upgrading" },
        { href: "/extensions", icon: Settings, label: "Extension Manager" },
        { href: "/training", icon: Zap, label: "Training System" },
      ]
    },
    {
      title: "Security",
      items: [
        { href: "/security", icon: Lock, label: "Quantum Security" },
        { href: "/firewall", icon: Shield, label: "Firewall AI" },
      ]
    },
    {
      title: "System",
      items: [
        { href: "/settings", icon: Settings, label: "Settings" },
        { href: "/status", icon: Timer, label: "System Status" },
      ]
    }
  ];
  
  return (
    <>
      {/* Mobile menu button - only visible on small screens */}
      <button 
        className="fixed top-4 left-4 z-50 p-2 bg-white rounded-md shadow-md lg:hidden"
        onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
      >
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-neutral-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
        </svg>
      </button>
      
      {/* Sidebar */}
      <aside 
        className={cn(
          "bg-white shadow-md w-64 flex-shrink-0 border-r border-neutral-200 overflow-y-auto",
          "fixed inset-y-0 left-0 z-40 transform transition-transform duration-300 ease-in-out lg:relative lg:translate-x-0",
          isMobileMenuOpen ? "translate-x-0" : "-translate-x-full"
        )}
      >
        {/* Logo and App Title */}
        <div className="px-6 py-4 border-b border-neutral-200">
          <h1 className="text-2xl font-bold text-primary-700 flex items-center">
            <Brain className="h-8 w-8 mr-2" />
            NeurAI
          </h1>
        </div>
        
        {/* Navigation */}
        <nav className="mt-2 px-3">
          {sections.map((section) => (
            <div className="mb-3" key={section.title}>
              <p className="text-neutral-500 text-xs font-semibold tracking-wider uppercase px-3 py-2">
                {section.title}
              </p>
              <ul>
                {section.items.map((item) => (
                  <NavItem
                    key={item.href}
                    href={item.href}
                    icon={item.icon}
                    label={item.label}
                    isActive={location === item.href}
                  />
                ))}
              </ul>
            </div>
          ))}
        </nav>
        
        {/* User profile section */}
        <div className="mt-auto px-5 py-4 border-t border-neutral-200">
          <div className="flex items-center">
            <div className="h-8 w-8 rounded-full bg-primary-500 flex items-center justify-center text-white font-semibold text-sm">
              {user?.username?.slice(0, 2).toUpperCase() || '??'}
            </div>
            <div className="ml-3">
              <p className="text-sm font-medium text-neutral-900">{user?.username || 'User'}</p>
              <p className="text-xs text-neutral-500">{user?.isAdmin ? 'Admin' : 'User'}</p>
            </div>
            <button 
              className="ml-auto text-neutral-400 hover:text-neutral-600"
              onClick={handleLogout}
              title="Logout"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
              </svg>
            </button>
          </div>
        </div>
      </aside>
      
      {/* Overlay for mobile */}
      {isMobileMenuOpen && (
        <div 
          className="fixed inset-0 bg-neutral-900/50 z-30 lg:hidden"
          onClick={() => setIsMobileMenuOpen(false)}
        />
      )}
    </>
  );
}
