import { NextResponse } from 'next/server';
import axios from 'axios';

// Configure the backend URL - change this to your actual backend URL
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000'; 

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { message, session_id } = body;

    if (!message) {
      return NextResponse.json(
        { error: 'Message is required' },
        { status: 400 }
      );
    }

    // Connect to the FastAPI backend
    try {
      // Convert session_id to string and provide fallback
      const sessionIdStr = typeof session_id === 'string' ? session_id : '';
      
      const payload = {
        message,
        session_id: sessionIdStr
      };
      
      console.log("Sending payload to backend:", payload);
      
      const response = await axios.post(`${BACKEND_URL}/api/chat`, payload, {
        headers: {
          'Content-Type': 'application/json',
        },
        timeout: 60000, // 60 second timeout - longer for tool processing
      });

      console.log("Received response from backend:", response.data);

      return NextResponse.json({
        message: response.data.message,
        references: response.data.references || [],
        session_id: response.data.session_id
      });
    } catch (error: any) {
      console.error('Error connecting to backend:', error.message);
      console.error('Error details:', error.response?.data || error);
      
      // Fallback to local response if backend is unavailable
      return NextResponse.json({
        message: "I'm currently experiencing connectivity issues with my knowledge database. Please try again later or ask a different question.",
        references: [],
        session_id: session_id || "",
        error: error.message,
      });
    }
  } catch (error: any) {
    console.error('API route error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
} 