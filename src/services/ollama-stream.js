// This is a replacement stream processor specifically for Ollama
// It handles Ollama's streaming format and converts it to OpenAI format

/**
 * Creates a proper streaming processor for Ollama responses
 */
export function createOllamaStreamProcessor(stream) {
  if (!stream) {
    throw new Error("Stream is null or undefined");
  }

  return (async function* () {
    const reader = stream.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    
    // Flag to track first chunk for role inclusion
    let isFirstChunk = true;
    
    try {
      while (true) {
        // Read chunk from stream
        const { done, value } = await reader.read();
        if (done) break;
        
        // Decode binary chunk to text
        const chunk = decoder.decode(value, { stream: true });
        buffer += chunk;
        
        // Process each complete JSON object (delimited by newlines)
        let lineEnd = buffer.indexOf('\n');
        while (lineEnd !== -1) {
          // Extract and parse one line
          const line = buffer.substring(0, lineEnd).trim();
          buffer = buffer.substring(lineEnd + 1);
          
          if (!line) continue;
          
          try {
            // Parse Ollama's JSON format
            const ollamaResponse = JSON.parse(line);
            
            // Extract content from Ollama format
            if (ollamaResponse.message && ollamaResponse.message.content) {
              const content = ollamaResponse.message.content;
              
              // Convert to OpenAI format
              // First chunk includes role, subsequent chunks only content
              const delta = isFirstChunk
                ? { role: 'assistant', content }
                : { content };
              
              // Create compatible chunk in OpenAI format
              const openaiChunk = {
                id: `ollama-${Date.now()}`,
                object: 'chat.completion.chunk',
                created: Math.floor(Date.now() / 1000),
                model: ollamaResponse.model || 'ollama-model',
                choices: [{
                  index: 0,
                  delta,
                  finish_reason: ollamaResponse.done ? 'stop' : null
                }]
              };
              
              // Only first chunk gets the role
              if (isFirstChunk) {
                isFirstChunk = false;
              }
              
              // Yield the converted chunk
              yield openaiChunk;
            }
          } catch (e) {
            // Skip invalid JSON
            console.error("Error processing Ollama stream:", e.message);
          }
          
          lineEnd = buffer.indexOf('\n');
        }
      }
    } finally {
      reader.releaseLock();
    }
  })();
}