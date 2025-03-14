// A specialized client for handling Ollama requests without OpenAI compatibility concerns
import { fetch } from 'undici';

// Makes a direct API call to Ollama and returns the raw stream
export async function sendOllamaRequest(baseURL, model, messages, options = {}) {
  console.log(`OLLAMA CLIENT: Sending request to ${baseURL}/api/chat`);
  
  // Create request body in Ollama format
  // Filter out any empty or null messages
  const filteredMessages = messages.filter(m => m && m.content);
  
  // Create a clean and minimal request body
  const requestBody = {
    model: model,
    messages: filteredMessages,
    stream: true,
    options: {
      temperature: options.temperature || 0.7,
      num_predict: options.max_tokens || 4096,
    }
  };
  
  console.log(`OLLAMA CLIENT: Request body:`, JSON.stringify(requestBody).substring(0, 200));
  
  // Make the API call
  try {
    const response = await fetch(`${baseURL}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });
    
    console.log(`OLLAMA CLIENT: Response status: ${response.status}`);
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Ollama API error: ${errorText}`);
    }
    
    // Clone the response to avoid "Body already used" errors
    const clonedResponse = response.clone();
    
    // Return the raw response body stream
    return clonedResponse.body;
  } catch (error) {
    console.error(`OLLAMA CLIENT ERROR: ${error.message}`);
    throw error;
  }
}

// Process the Ollama stream format directly to extract content as it arrives
export function processOllamaStream(stream) {
  return (async function*() {
    if (!stream) {
      throw new Error("Stream is null or undefined");
    }
    
    const reader = stream.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    
    try {
      while (true) {
        // Read raw chunk
        const { done, value } = await reader.read();
        if (done) break;
        
        // Convert to text
        const chunk = decoder.decode(value, { stream: true });
        buffer += chunk;
        
        // Process each line (Ollama sends newline-delimited JSON)
        let lineEnd = buffer.indexOf('\n');
        while (lineEnd !== -1) {
          const line = buffer.substring(0, lineEnd).trim();
          buffer = buffer.substring(lineEnd + 1);
          
          if (!line) continue;
          
          try {
            // Parse the JSON response
            const data = JSON.parse(line);
            
            // Extract content if available
            if (data.message && data.message.content) {
              yield {
                content: data.message.content,
                done: data.done || false
              };
            }
            
            // If this is the final message, handle as needed
            if (data.done) {
              console.log("OLLAMA CLIENT: Stream complete");
            }
          } catch (e) {
            console.error(`OLLAMA CLIENT: Error parsing JSON: ${e.message}`);
          }
          
          lineEnd = buffer.indexOf('\n');
        }
      }
    } finally {
      reader.releaseLock();
    }
  })();
}

// Send a request to Ollama and process the stream to get content
export async function getOllamaCompletion(baseURL, model, messages, options = {}) {
  try {
    // Log the model being used
    console.log(`OLLAMA: Using model ${model}`);
    
    // Get the raw stream
    const stream = await sendOllamaRequest(baseURL, model, messages, options);
    
    // Process it to get a stream of content chunks
    return processOllamaStream(stream);
  } catch (error) {
    console.error(`OLLAMA CLIENT: getOllamaCompletion error: ${error.message}`);
    throw error;
  }
}