import { MainLayout } from "@/components/layout/MainLayout";
import { ChatPanel } from "@/components/ai/ChatPanel";
import { AdminDashboard } from "@/components/ai/AdminDashboard";
import { useEffect } from "react";
import { useAuth } from "@/hooks/use-auth";
import { connectWebSocket } from "@/lib/websocket";

export default function Dashboard() {
  const { user } = useAuth();
  
  // Initialize WebSocket connection when dashboard mounts
  useEffect(() => {
    if (user) {
      connectWebSocket(user.id);
    }
  }, [user]);
  
  return (
    <MainLayout>
      <ChatPanel />
      <AdminDashboard />
    </MainLayout>
  );
}
