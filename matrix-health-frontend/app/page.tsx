'use client';

import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';

type MessageType = {
  id: string;
  content: string;
  author: 'user' | 'ai' | 'system';
  timestamp: Date;
};

const MatrixTerminal = () => {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<MessageType[]>([
    {
      id: '1',
      content: 'NEURAL HEALTH INTERFACE INITIALIZED',
      author: 'system',
      timestamp: new Date(),
    },
    {
      id: '2',
      content: 'How may I assist with your mental well-being today?',
      author: 'ai',
      timestamp: new Date(),
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [isWaitingForResponse, setIsWaitingForResponse] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const consoleRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Load session ID from localStorage
  useEffect(() => {
    const savedSessionId = localStorage.getItem('matrixSessionId');
    if (savedSessionId) {
      setSessionId(savedSessionId);
    }
  }, []);

  // Automatically scroll to the bottom when new messages are added
  useEffect(() => {
    if (consoleRef.current) {
      consoleRef.current.scrollTop = consoleRef.current.scrollHeight;
    }
  }, [messages]);

  // Focus input on load
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.focus();
    }
  }, []);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInput(e.target.value);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!input.trim() || isWaitingForResponse) return;
    
    const userMessage: MessageType = {
      id: Date.now().toString(),
      content: input,
      author: 'user',
      timestamp: new Date(),
    };
    
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsWaitingForResponse(true);
    setIsLoading(true);
    
    // Add a system message to show processing
    setMessages((prev) => [
      ...prev,
      {
        id: `${Date.now()}-system`,
        content: 'PROCESSING REQUEST',
        author: 'system',
        timestamp: new Date(),
      },
    ]);

    try {
      // Call the API endpoint with session ID if available
      const response = await axios.post('/api/chat', {
        message: input,
        session_id: sessionId
      });
      
      // Remove the processing message
      setMessages((prev) => 
        prev.filter(msg => msg.id !== `${userMessage.id}-system`)
      );
      
      // Save the session ID for future requests
      if (response.data.session_id) {
        setSessionId(response.data.session_id);
        localStorage.setItem('matrixSessionId', response.data.session_id);
      }
      
      // Add the AI response
      setMessages((prev) => [
        ...prev,
        {
          id: `${Date.now()}-ai`,
          content: response.data.message,
          author: 'ai',
          timestamp: new Date(),
        },
      ]);
      
      // If there are references, add them as a system message
      if (response.data.references && response.data.references.length > 0) {
        setMessages((prev) => [
          ...prev,
          {
            id: `${Date.now()}-system-refs`,
            content: `REFERENCES: ${response.data.references.join(' • ')}`,
            author: 'system',
            timestamp: new Date(),
          },
        ]);
      }
      
      setIsLoading(false);
      setIsWaitingForResponse(false);
    } catch (error) {
      console.error('Error sending message:', error);
      
      setMessages((prev) => [
        ...prev,
        {
          id: `${Date.now()}-error`,
          content: 'CONNECTION ERROR: Unable to process request',
          author: 'system',
          timestamp: new Date(),
        },
      ]);
      
      setIsLoading(false);
      setIsWaitingForResponse(false);
    }
  };

  const renderMessage = (message: MessageType) => {
    const authorClassMap = {
      user: 'matrix-user',
      ai: 'matrix-ai',
      system: 'matrix-system',
    };

    const authorPrefixMap = {
      user: '> USER:',
      ai: '> NEURAL:',
      system: '> SYS:',
    };

    return (
      <motion.div
        key={message.id}
        className={`matrix-message ${authorClassMap[message.author]}`}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        <span className="font-bold">{authorPrefixMap[message.author]} </span>
        {message.author === 'ai' || message.author === 'system' ? (
          <ReactMarkdown>{message.content}</ReactMarkdown>
        ) : (
          message.content
        )}
        {message.author === 'system' && message.content === 'PROCESSING REQUEST' && (
          <span className="loading-dots"></span>
        )}
      </motion.div>
    );
  };

  return (
    <main className="flex-1 flex flex-col">
      <div className="matrix-console" ref={consoleRef}>
        {messages.map(renderMessage)}
        {isLoading && (
          <motion.div
            className="matrix-message matrix-system"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.3 }}
          >
            <span className="blink">_</span>
          </motion.div>
        )}
        </div>
      
      <form onSubmit={handleSubmit} className="matrix-input-container">
        <span className="matrix-input-prefix">QUERY&gt;</span>
        <input
          type="text"
          value={input}
          onChange={handleInputChange}
          className="matrix-input"
          placeholder="Enter your message..."
          disabled={isWaitingForResponse}
          ref={inputRef}
        />
        <button
          type="submit"
          className={`px-4 py-2 ${
            isWaitingForResponse
              ? 'opacity-50 cursor-not-allowed'
              : 'hover:bg-green-900 hover:bg-opacity-30'
          }`}
          disabled={isWaitingForResponse}
        >
          SEND
        </button>
      </form>
    </main>
  );
};

export default function Home() {
  return <MatrixTerminal />;
}
