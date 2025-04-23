import { AIMessage, WebSocketMessage } from "./types";

let socket: WebSocket | null = null;
let reconnectAttempts = 0;
const maxReconnectAttempts = 5;
const reconnectDelay = 2000;

type MessageHandler = (message: WebSocketMessage) => void;
const messageHandlers: MessageHandler[] = [];

export function connectWebSocket(userId?: number): WebSocket {
  // If socket is already open, return it
  if (socket && socket.readyState === WebSocket.OPEN) {
    return socket;
  }
  
  // Close existing socket if not already closed
  if (socket && socket.readyState !== WebSocket.CLOSED) {
    socket.close();
  }

  // Determine the correct protocol and URL
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const host = window.location.host;
  const wsUrl = `${protocol}//${host}/ws`;
  
  // Create new socket
  socket = new WebSocket(wsUrl);
  
  socket.onopen = () => {
    console.log("WebSocket connected successfully");
    reconnectAttempts = 0;
    
    // Send authentication if user is logged in
    if (userId) {
      // Use a small delay to ensure socket is fully ready
      setTimeout(() => {
        sendMessage({
          type: 'auth',
          data: { userId }
        });
      }, 100);
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
    socket = null;
    
    // Attempt to reconnect if not a normal closure
    if (event.code !== 1000 && event.code !== 1001) {
      tryReconnect();
    }
  };
  
  socket.onerror = (error) => {
    console.error("WebSocket error:", error);
    // Don't set socket to null here, let onclose handle it
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

export function sendMessage(message: WebSocketMessage): boolean {
  // Check if socket exists
  if (!socket) {
    console.warn("No WebSocket connection exists. Attempting to connect...");
    socket = connectWebSocket();
    
    // Queue message to be sent once connected
    const queuedMessage = { ...message };
    setTimeout(() => {
      if (socket && socket.readyState === WebSocket.OPEN) {
        console.log("Sending queued message");
        socket.send(JSON.stringify(queuedMessage));
      } else {
        console.error("Failed to send queued message: WebSocket not ready");
      }
    }, 500);
    
    return false;
  }
  
  // Check socket state
  if (socket.readyState === WebSocket.CONNECTING) {
    console.log("WebSocket is still connecting. Queueing message...");
    
    // Queue message to be sent once connected
    const queuedMessage = { ...message };
    setTimeout(() => {
      if (socket && socket.readyState === WebSocket.OPEN) {
        console.log("Sending queued message");
        socket.send(JSON.stringify(queuedMessage));
      } else {
        console.error("Failed to send queued message: WebSocket not ready");
      }
    }, 500);
    
    return false;
  } else if (socket.readyState !== WebSocket.OPEN) {
    console.error(`WebSocket is not open (state: ${socket.readyState}). Reconnecting...`);
    socket = connectWebSocket();
    return false;
  }
  
  // Socket is ready, send message
  try {
    socket.send(JSON.stringify(message));
    return true;
  } catch (error) {
    console.error("Error sending WebSocket message:", error);
    return false;
  }
}

export function sendChatMessage(message: AIMessage): boolean {
  console.log('Attempting to send chat message via WebSocket', message.conversationId);
  
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    console.warn('WebSocket not connected when trying to send chat message. Creating new connection...');
    
    // Try to create a new connection
    try {
      socket = connectWebSocket(message.userId as number);
      
      // Queue the message to be sent after connection is established
      setTimeout(() => {
        if (socket && socket.readyState === WebSocket.OPEN) {
          console.log('Sending delayed chat message after reconnection');
          socket.send(JSON.stringify({
            type: 'chat-message',
            message
          }));
        } else {
          console.error('Failed to send delayed chat message - socket still not ready');
        }
      }, 1000); // Wait for connection to establish
      
      return false;
    } catch (error) {
      console.error('Failed to reconnect WebSocket for sending message', error);
      return false;
    }
  }
  
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
