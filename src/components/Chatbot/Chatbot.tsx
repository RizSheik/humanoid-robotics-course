/**
 * RAG Chatbot Component for Docusaurus
 * 
 * This component implements the RAG (Retrieval Augmented Generation) chatbot UI
 * that allows users to ask questions about the educational content.
 */

import React, { useState, useRef, useEffect } from 'react';
import Layout from '@theme/Layout';
import clsx from 'clsx';

const ChatbotComponent = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const messagesEndRef = useRef(null);

  // Initialize session on component mount
  useEffect(() => {
    const initSession = async () => {
      try {
        // Get or create session
        const response = await fetch('/api/session', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          }
        });
        const sessionData = await response.json();
        setSessionId(sessionData.sessionId);
      } catch (error) {
        console.error('Error initializing session:', error);
      }
    };

    initSession();
  }, []);

  // Scroll to bottom of messages
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputValue.trim() || !sessionId) return;

    // Add user message to chat
    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: inputValue,
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Send query to backend API
      const response = await fetch('/api/chat/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query: inputValue,
          session_id: sessionId
        })
      });

      const data = await response.json();

      // Add bot response to chat
      const botMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: data.response,
        sources: data.sources, // Include sources if available
        timestamp: new Date()
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={clsx('container margin-vert--lg')}>
      <div className="row">
        <div className="col col--8 col--offset-2">
          <h2>Course Assistant</h2>
          <p>Ask questions about the course content and get AI-powered answers based on the educational material.</p>
          
          <div className="card">
            <div className="card__body">
              {/* Chat container */}
              <div className="chat-container" style={{ height: '400px', overflowY: 'auto', marginBottom: '1rem', padding: '1rem', border: '1px solid #ddd', borderRadius: '4px' }}>
                {messages.length === 0 ? (
                  <div className="text--center padding--md">
                    <p>Ask me anything about the course content!</p>
                    <small>I can help explain concepts, find relevant sections, and answer questions based on the material.</small>
                  </div>
                ) : (
                  <div className="chat-messages">
                    {messages.map((message) => (
                      <div key={message.id} className={`margin-bottom--sm ${message.role === 'user' ? 'text--right' : ''}`}>
                        <div 
                          className={clsx(
                            'padding--sm', 
                            'radius--sm',
                            message.role === 'user' 
                              ? 'background-color--primary text--white' 
                              : 'background-color--gray'
                          )}
                          style={{ 
                            maxWidth: '80%', 
                            display: 'inline-block',
                            textAlign: 'left'
                          }}
                        >
                          <div>{message.content}</div>
                          {message.sources && message.sources.length > 0 && (
                            <small className="margin-top--xs display--block">
                              Sources: {message.sources.slice(0, 2).map(src => src.chapterTitle).join(', ')}
                            </small>
                          )}
                          <small className="margin-top--xs display--block">{message.timestamp.toLocaleTimeString()}</small>
                        </div>
                      </div>
                    ))}
                    {isLoading && (
                      <div className="margin-bottom--sm">
                        <div className={clsx('padding--sm', 'radius--sm', 'background-color--gray')} style={{ maxWidth: '80%', display: 'inline-block' }}>
                          <em>Thinking...</em>
                        </div>
                      </div>
                    )}
                    <div ref={messagesEndRef} />
                  </div>
                )}
              </div>

              {/* Input form */}
              <form onSubmit={handleSubmit}>
                <div className="input-group">
                  <input
                    type="text"
                    className="input-group__field"
                    placeholder="Ask a question about the course..."
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    disabled={isLoading}
                  />
                  <button 
                    type="submit" 
                    className="button button--primary input-group__button"
                    disabled={isLoading || !inputValue.trim()}
                  >
                    {isLoading ? 'Sending...' : 'Send'}
                  </button>
                </div>
                <div className="margin-top--sm">
                  <small>Examples: "Explain forward kinematics", "What are the key components of a digital twin?"</small>
                </div>
              </form>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default function ChatbotPage() {
  return (
    <Layout title="Course Assistant" description="Ask questions about the course content">
      <ChatbotComponent />
    </Layout>
  );
}