import { AIMessage, WebSocketMessage } from "./types";

let socket: WebSocket | null = null;
let reconnectAttempts = 0;
const maxReconnectAttempts = 5;
const reconnectDelay = 2000;

type MessageHandler = (message: WebSocketMessage) => void;
const messageHandlers: MessageHandler[] = [];

export function connectWebSocket(userId?: number): WebSocket {
  if (socket && socket.readyState === WebSocket.OPEN) {
    return socket;
  }

  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws`;
  
  socket = new WebSocket(wsUrl);
  
  socket.onopen = () => {
    console.log("WebSocket connected");
    reconnectAttempts = 0;
    
    // Send authentication if user is logged in
    if (userId) {
      sendMessage({
        type: 'auth',
        data: { userId }
      });
    }
  };
  
  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data) as WebSocketMessage;
      
      // Notify all registered handlers
      messageHandlers.forEach(handler => handler(data));
      
    } catch (error) {
      console.error("Failed to parse WebSocket message:", error);
    }
  };
  
  socket.onclose = (event) => {
    console.log(`WebSocket closed: ${event.code} ${event.reason}`);
    
    // Attempt to reconnect if not a normal closure
    if (event.code !== 1000 && event.code !== 1001) {
      tryReconnect();
    }
  };
  
  socket.onerror = (error) => {
    console.error("WebSocket error:", error);
  };
  
  return socket;
}

function tryReconnect() {
  if (reconnectAttempts >= maxReconnectAttempts) {
    console.error("Max reconnect attempts reached. Please refresh the page.");
    return;
  }
  
  reconnectAttempts++;
  
  setTimeout(() => {
    console.log(`Attempting to reconnect (${reconnectAttempts}/${maxReconnectAttempts})...`);
    connectWebSocket();
  }, reconnectDelay * reconnectAttempts);
}

export function sendMessage(message: WebSocketMessage) {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    console.error("WebSocket is not connected. Cannot send message.");
    return false;
  }
  
  socket.send(JSON.stringify(message));
  return true;
}

export function sendChatMessage(message: AIMessage) {
  return sendMessage({
    type: 'chat-message',
    message
  });
}

export function registerMessageHandler(handler: MessageHandler) {
  messageHandlers.push(handler);
  
  // Return function to unregister
  return () => {
    const index = messageHandlers.indexOf(handler);
    if (index !== -1) {
      messageHandlers.splice(index, 1);
    }
  };
}

export function closeWebSocket() {
  if (socket) {
    socket.close(1000, "Normal closure");
    socket = null;
  }
}
