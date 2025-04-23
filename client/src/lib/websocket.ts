import { AIMessage, WebSocketMessage } from "./types";
import { apiRequest } from "./queryClient";

let socket: WebSocket | null = null;
let reconnectAttempts = 0;
const maxReconnectAttempts = 5;
const reconnectDelay = 2000;
let reconnectTimer: NodeJS.Timeout | null = null;
let messageQueue: WebSocketMessage[] = [];
let isProcessingQueue = false;
let retryCount = 0;
const maxRetryCount = 3;

type MessageHandler = (message: WebSocketMessage) => void;
const messageHandlers: MessageHandler[] = [];

export function connectWebSocket(userId?: number): WebSocket {
  // If socket is already open, return it
  if (socket && socket.readyState === WebSocket.OPEN) {
    console.log("Reusing existing WebSocket connection");
    return socket;
  }
  
  // Clear any pending reconnect timer
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
  
  // Close existing socket if not already closed
  if (socket && socket.readyState !== WebSocket.CLOSED) {
    console.log("Closing existing socket before creating a new one");
    socket.onclose = null; // Prevent triggering reconnect logic when we're closing intentionally
    socket.close();
  }

  console.log("Creating new WebSocket connection");
  
  // Determine the correct protocol and URL
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const host = window.location.host;
  
  // Construct URL carefully, ensuring we're not using undefined port
  let wsUrl = `${protocol}//${host}/ws`;
  
  // Additional logging for debugging
  console.log(`Constructing WebSocket URL: ${wsUrl}`);
  
  try {
    // Create new socket with error handling
    socket = new WebSocket(wsUrl);
  } catch (error) {
    console.error("Error creating WebSocket connection:", error);
    tryReconnect(userId);
    return socket;
  }
  
  socket.onopen = () => {
    console.log("WebSocket connected successfully");
    reconnectAttempts = 0;
    
    // Send authentication if user is logged in
    if (userId) {
      console.log("Sending authentication");
      // Use a small delay to ensure socket is fully ready
      setTimeout(() => {
        if (socket && socket.readyState === WebSocket.OPEN) {
          sendMessage({
            type: 'auth',
            data: { userId }
          });
          
          // Process any queued messages
          if (messageQueue.length > 0) {
            console.log(`Processing ${messageQueue.length} queued messages`);
            processMessageQueue();
          }
        }
      }, 200);
    }
  };
  
  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data) as WebSocketMessage;
      
      // Notify all registered handlers
      messageHandlers.forEach(handler => {
        try {
          handler(data);
        } catch (handlerError) {
          console.error("Error in message handler:", handlerError);
        }
      });
      
    } catch (error) {
      console.error("Failed to parse WebSocket message:", error);
    }
  };
  
  socket.onclose = (event) => {
    console.log(`WebSocket closed: ${event.code} ${event.reason || ''}`);
    socket = null;
    
    // Attempt to reconnect if not a normal closure
    if (event.code !== 1000 && event.code !== 1001) {
      tryReconnect(userId);
    }
  };
  
  socket.onerror = (error) => {
    console.error("WebSocket error:", error);
    // Don't set socket to null here, let onclose handle it
  };
  
  return socket;
}

function tryReconnect(userId?: number) {
  if (reconnectAttempts >= maxReconnectAttempts) {
    console.error("Max reconnect attempts reached. Please refresh the page.");
    return;
  }
  
  // Clear any existing timer
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
  }
  
  reconnectAttempts++;
  const delay = reconnectDelay * reconnectAttempts;
  
  console.log(`Scheduling reconnect in ${delay}ms (attempt ${reconnectAttempts}/${maxReconnectAttempts})...`);
  
  reconnectTimer = setTimeout(() => {
    console.log(`Attempting to reconnect (${reconnectAttempts}/${maxReconnectAttempts})...`);
    connectWebSocket(userId);
  }, delay);
}

export function sendMessage(message: WebSocketMessage): boolean {
  // Check if socket exists
  if (!socket) {
    console.warn("No WebSocket connection exists. Queueing message.");
    messageQueue.push(message);
    return false;
  }
  
  // Check socket state
  if (socket.readyState === WebSocket.CONNECTING) {
    console.log("WebSocket is still connecting. Queueing message...");
    messageQueue.push(message);
    return false;
  } else if (socket.readyState !== WebSocket.OPEN) {
    console.error(`WebSocket is not open (state: ${socket.readyState}). Queueing message and reconnecting...`);
    messageQueue.push(message);
    return false;
  }
  
  // Socket is ready, send message
  try {
    socket.send(JSON.stringify(message));
    return true;
  } catch (error) {
    console.error("Error sending WebSocket message:", error);
    messageQueue.push(message);
    return false;
  }
}

export function sendChatMessage(message: AIMessage): boolean {
  console.log('Attempting to send chat message via WebSocket', message.conversationId);
  
  // Create the standard message format
  const formattedMessage: WebSocketMessage = {
    type: 'chat-message',
    message
  };
  
  // Check if socket is ready
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    console.warn('WebSocket not connected when trying to send chat message');
    
    // Store in queue for later sending
    messageQueue.push(formattedMessage);
    
    // Try to create a new connection if we have a user ID
    if (message.userId) {
      try {
        connectWebSocket(message.userId as number);
      } catch (error) {
        console.error('Failed to connect WebSocket for sending message', error);
      }
    }
    
    return false;
  }
  
  // Send using the standard function which handles errors
  return sendMessage(formattedMessage);
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

// Process any queued messages with fallback to REST API
async function processMessageQueue() {
  let isProcessing = true;
  let retryCount = 0;
  const maxRetryCount = 3;
  
  while (messageQueue.length > 0 && isProcessing) {
    const message = messageQueue[0]; // Peek at the first message
    
    // Try WebSocket first
    if (socket && socket.readyState === WebSocket.OPEN) {
      try {
        socket.send(JSON.stringify(message));
        messageQueue.shift(); // Remove from queue if successful
        retryCount = 0;
      } catch (error) {
        console.error("Error sending queued message via WebSocket:", error);
        retryCount++;
        
        if (retryCount >= maxRetryCount) {
          // Give up on this message after max retries
          messageQueue.shift();
          retryCount = 0;
          console.error("Max retries reached, discarding message");
        } else {
          // Stop processing for now, will retry later
          isProcessing = false;
        }
      }
    } 
    // If WebSocket is not available, try REST API fallback for chat messages
    else if (message.type === 'chat-message' && message.message) {
      try {
        // Use REST API as fallback
        const response = await apiRequest("POST", "/api/chat", {
          message: message.message
        });
        
        // Handle response
        if (response.ok) {
          const data = await response.json();
          messageQueue.shift(); // Remove from queue
          
          // Manually dispatch response to handlers if successful
          if (data.aiResponse) {
            messageHandlers.forEach(handler => {
              try {
                handler({
                  type: 'new-message',
                  message: data.aiResponse
                });
              } catch (handlerError) {
                console.error("Error in REST API fallback message handler:", handlerError);
              }
            });
          }
        } else {
          retryCount++;
          if (retryCount >= maxRetryCount) {
            messageQueue.shift();
            retryCount = 0;
          }
          isProcessing = false;
        }
      } catch (error) {
        console.error("Error sending message via REST API fallback:", error);
        retryCount++;
        
        if (retryCount >= maxRetryCount) {
          messageQueue.shift();
          retryCount = 0;
        }
        isProcessing = false;
      }
    } 
    // For non-chat messages with no WebSocket, we can't process them (auth, etc)
    else {
      // Can't process non-chat messages without WebSocket
      // Keep them in queue for next connection attempt
      isProcessing = false;
    }
  }
  
  // If there are more messages and we stopped processing due to errors,
  // retry after a delay
  if (messageQueue.length > 0 && !isProcessing) {
    setTimeout(processMessageQueue, 3000);
  }
}

export function closeWebSocket() {
  if (socket) {
    socket.close(1000, "Normal closure");
    socket = null;
  }
  
  // Try to process any remaining messages via REST API if possible
  if (messageQueue.length > 0) {
    processMessageQueue();
  }
}
