import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { MemoryItem } from "./MemoryItem";
import { AIStatus, FeatureFlags, AIMemory } from "@/lib/types";
import { CustomCheckbox } from "@/components/ui/custom-checkbox";
import { Button } from "@/components/ui/button";
import { 
  Plus, 
  RefreshCw, 
  FileText, 
  Shield, 
  Zap 
} from "lucide-react";
import { apiRequest } from "@/lib/queryClient";

export function AdminDashboard() {
  // AI System Status
  const [aiStatus, setAiStatus] = useState<AIStatus>({
    llama3: true,
    gemma3: true,
    neuroSymbolic: true,
    memory: true,
    systemLoad: 42
  });
  
  // Feature flags
  const [featureFlags, setFeatureFlags] = useState<FeatureFlags>({
    selfUpgradeEnabled: true,
    codeExecutionEnabled: true,
    autonomousActionsEnabled: false,
    internetAccessEnabled: true
  });
  
  // Fetch recent memories
  const { data: memories, isLoading: isLoadingMemories } = useQuery<AIMemory[]>({
    queryKey: ['/api/memories'],
    queryFn: async () => {
      const res = await apiRequest('GET', '/api/memories?limit=3');
      return await res.json();
    }
  });
  
  // Update feature flags
  const handleFeatureFlagChange = (flag: keyof FeatureFlags) => {
    setFeatureFlags(prev => ({
      ...prev,
      [flag]: !prev[flag]
    }));
    
    // In a real implementation, we would make an API call to update the setting
    // apiRequest('PUT', `/api/settings/featureFlags`, { 
    //   value: { ...featureFlags, [flag]: !featureFlags[flag] } 
    // });
  };
  
  return (
    <div className="hidden lg:block lg:w-1/3 overflow-y-auto bg-white border-l border-neutral-200">
      {/* Dashboard Header */}
      <div className="bg-white h-12 border-b border-neutral-200 flex items-center px-4">
        <h2 className="text-lg font-medium">Admin Dashboard</h2>
        <button className="ml-auto text-neutral-500 hover:text-neutral-700 p-1 rounded-full hover:bg-neutral-100">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16m-7 6h7" />
          </svg>
        </button>
      </div>
      
      {/* Dashboard Content */}
      <div className="p-4">
        {/* AI Status Card */}
        <div className="bg-neutral-50 rounded-lg p-4 mb-4 shadow-sm">
          <h3 className="text-sm font-semibold text-neutral-700 mb-3">AI System Status</h3>
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-neutral-600">Llama3 Model</span>
            <span className="px-2 py-1 bg-green-500 text-white text-xs rounded-full">{aiStatus.llama3 ? 'Active' : 'Inactive'}</span>
          </div>
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-neutral-600">Gemma3 Model</span>
            <span className="px-2 py-1 bg-green-500 text-white text-xs rounded-full">{aiStatus.gemma3 ? 'Active' : 'Inactive'}</span>
          </div>
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-neutral-600">Neuro-Symbolic</span>
            <span className="px-2 py-1 bg-green-500 text-white text-xs rounded-full">{aiStatus.neuroSymbolic ? 'Active' : 'Inactive'}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-neutral-600">Memory System</span>
            <span className="px-2 py-1 bg-green-500 text-white text-xs rounded-full">{aiStatus.memory ? 'Active' : 'Inactive'}</span>
          </div>
          <div className="mt-3 pt-3 border-t border-neutral-200">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-neutral-700">System Load</span>
              <span className="text-xs text-neutral-700">{aiStatus.systemLoad}%</span>
            </div>
            <div className="w-full bg-neutral-200 rounded-full h-1.5 mt-1">
              <div 
                className="bg-primary-600 h-1.5 rounded-full" 
                style={{ width: `${aiStatus.systemLoad}%` }}
              ></div>
            </div>
          </div>
        </div>
        
        {/* Quick Actions */}
        <div className="bg-white border border-neutral-200 rounded-lg p-4 mb-4 shadow-sm">
          <h3 className="text-sm font-semibold text-neutral-700 mb-3">Quick Actions</h3>
          <div className="grid grid-cols-2 gap-2">
            <button className="flex flex-col items-center justify-center bg-neutral-50 hover:bg-neutral-100 rounded-md p-3 transition-colors duration-200">
              <Plus className="h-6 w-6 text-primary-600 mb-1" />
              <span className="text-xs text-neutral-700">Add Extension</span>
            </button>
            <button className="flex flex-col items-center justify-center bg-neutral-50 hover:bg-neutral-100 rounded-md p-3 transition-colors duration-200">
              <RefreshCw className="h-6 w-6 text-primary-600 mb-1" />
              <span className="text-xs text-neutral-700">Update Models</span>
            </button>
            <button className="flex flex-col items-center justify-center bg-neutral-50 hover:bg-neutral-100 rounded-md p-3 transition-colors duration-200">
              <FileText className="h-6 w-6 text-primary-600 mb-1" />
              <span className="text-xs text-neutral-700">View Logs</span>
            </button>
            <button className="flex flex-col items-center justify-center bg-neutral-50 hover:bg-neutral-100 rounded-md p-3 transition-colors duration-200">
              <Shield className="h-6 w-6 text-primary-600 mb-1" />
              <span className="text-xs text-neutral-700">Secure System</span>
            </button>
          </div>
        </div>
        
        {/* AI Memory */}
        <div className="bg-white border border-neutral-200 rounded-lg p-4 mb-4 shadow-sm">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-neutral-700">Recent Memory</h3>
            <button className="text-xs text-primary-600 hover:text-primary-800">View All</button>
          </div>
          <div className="space-y-3">
            {isLoadingMemories ? (
              <div className="text-sm text-center py-3 text-neutral-500">Loading memories...</div>
            ) : memories && memories.length > 0 ? (
              memories.map((memory) => (
                <MemoryItem key={memory.id} memory={memory} />
              ))
            ) : (
              <div className="text-sm text-center py-3 text-neutral-500">No memories yet</div>
            )}
          </div>
        </div>
        
        {/* Feature Control */}
        <div className="bg-white border border-neutral-200 rounded-lg p-4 mb-4 shadow-sm">
          <h3 className="text-sm font-semibold text-neutral-700 mb-3">Feature Control</h3>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-medium text-neutral-700">Self-Upgrade Mode</p>
                <p className="text-xs text-neutral-500 mt-0.5">Allow AI to update its own code</p>
              </div>
              <CustomCheckbox 
                checked={featureFlags.selfUpgradeEnabled}
                onChange={() => handleFeatureFlagChange('selfUpgradeEnabled')}
              />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-medium text-neutral-700">Code Execution</p>
                <p className="text-xs text-neutral-500 mt-0.5">Allow AI to run code in sandbox</p>
              </div>
              <CustomCheckbox 
                checked={featureFlags.codeExecutionEnabled}
                onChange={() => handleFeatureFlagChange('codeExecutionEnabled')}
              />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-medium text-neutral-700">Autonomous Actions</p>
                <p className="text-xs text-neutral-500 mt-0.5">Allow AI to take actions without prompts</p>
              </div>
              <CustomCheckbox 
                checked={featureFlags.autonomousActionsEnabled}
                onChange={() => handleFeatureFlagChange('autonomousActionsEnabled')}
              />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-medium text-neutral-700">Internet Access</p>
                <p className="text-xs text-neutral-500 mt-0.5">Allow AI to access external resources</p>
              </div>
              <CustomCheckbox 
                checked={featureFlags.internetAccessEnabled}
                onChange={() => handleFeatureFlagChange('internetAccessEnabled')}
              />
            </div>
          </div>
        </div>
        
        {/* Current Implementation */}
        <div className="bg-primary-50 border border-primary-200 rounded-lg p-4 mb-4">
          <div className="flex items-start">
            <div className="flex-shrink-0 h-8 w-8 rounded-full bg-primary-100 flex items-center justify-center text-primary-700">
              <Zap className="h-5 w-5" />
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-semibold text-primary-700">Active Implementation</h3>
              <p className="text-xs text-primary-800 mt-1">Speech Recognition Integration (42% complete)</p>
              <div className="w-full bg-primary-100 rounded-full h-1.5 mt-2">
                <div className="bg-primary-600 h-1.5 rounded-full" style={{ width: '42%' }}></div>
              </div>
              <div className="mt-2 flex space-x-2">
                <Button className="px-2 py-1 h-auto bg-primary-600 text-white text-xs rounded hover:bg-primary-700">
                  View Details
                </Button>
                <Button variant="outline" className="px-2 py-1 h-auto bg-white text-primary-600 border border-primary-600 text-xs rounded hover:bg-primary-50">
                  Pause
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
