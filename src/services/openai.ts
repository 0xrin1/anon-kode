import { OpenAI } from "openai";
import { getGlobalConfig } from "../utils/config";
import { ProxyAgent, fetch } from 'undici';
import { createOllamaStreamProcessor } from './ollama-stream.js';
import { getOllamaCompletion } from './ollama-client.js';
import modelsList from '../constants/models';

// Helper function to get model config from model name
function getModelConfig(modelName: string) {
  // Look through all model categories
  for (const category in modelsList) {
    if (Object.prototype.hasOwnProperty.call(modelsList, category)) {
      const models = modelsList[category];
      // Find matching model
      const model = models.find(m => m.model === modelName);
      if (model) {
        return model;
      }
    }
  }
  return null;
}

export async function getCompletion(
  type: 'large' | 'small', 
  opts: OpenAI.ChatCompletionCreateParams
): Promise<OpenAI.ChatCompletion | AsyncIterable<OpenAI.ChatCompletionChunk>> {
  const config = getGlobalConfig()
  const apiKey = type === 'large' ? config.largeModelApiKey : config.smallModelApiKey
  const baseURL = type === 'large' ? config.largeModelBaseURL : config.smallModelBaseURL
  const proxy = config.proxy ? new ProxyAgent(config.proxy) : undefined
  
  // Check if this is Ollama based on URL, model name, or provider
  const isOllama = (baseURL && (baseURL.includes("ollama") || baseURL.includes("192.168.191.55"))) || 
                  (opts.model && (opts.model.includes("hf.co/") || opts.model.includes("ollama") || opts.model.includes("qwq"))) ||
                  (type === 'large' && config.largeModelProvider === "ollama") ||
                  (type === 'small' && config.smallModelProvider === "ollama");
                  
  console.log(`OLLAMA CHECK: isOllama=${isOllama}, baseURL=${baseURL}, model=${opts.model}, provider=${type === 'large' ? config.largeModelProvider : config.smallModelProvider}`);
  
  // Only log in verbose mode
  if (process.env.VERBOSE) {
    console.log(`API Request to ${baseURL} for model ${opts.model} (isOllama: ${isOllama})`);
  }
  
  // Prepare headers based on provider
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };
  
  // For Ollama, don't add Authorization header at all since it's not needed
  if (!isOllama && apiKey) {
    // Only add Authorization for non-Ollama providers
    headers['Authorization'] = `Bearer ${apiKey}`;
  }
  
  // Debug headers only in verbose mode
  if (process.env.VERBOSE) {
    console.log("Request headers:", JSON.stringify(headers));
  }
  
  if (opts.stream) {
    // Log the model being used 
    console.log(`Using model: ${opts.model}`);
    
    // *** FOR OLLAMA: USE COMPLETELY SEPARATE CLIENT PATH ***
    if (isOllama) {
      console.log("Using direct Ollama client");
      
      try {
        // Check if we have a custom system prompt for this model in the models config
      const modelConfig = getModelConfig(opts.model);
      const customSystemPrompt = modelConfig?.customSystemPrompt;
      
      // Create a copy of messages to potentially modify
      let modifiedMessages = [...opts.messages];
      
      // If we have a custom system prompt, replace all system messages with our custom one
      if (customSystemPrompt) {
        // Remove existing system messages
        modifiedMessages = modifiedMessages.filter(msg => msg.role !== 'system');
        
        // Add our custom system prompt at the beginning
        modifiedMessages.unshift({
          role: 'system',
          content: customSystemPrompt
        });
        
        console.log("Using custom system prompt for model:", opts.model);
      }
      
      // Use specialized Ollama client that bypasses OpenAI compatibility layers
      const ollamaStream = await getOllamaCompletion(
        baseURL, 
        opts.model, 
        modifiedMessages,
        {
          temperature: opts.temperature,
          max_tokens: opts.max_tokens
        }
      );
        
        // Accumulate full response for function call detection
        let accumulatedContent = '';
        const fullContent = [];
        
        // First pass: collect full content and check for function calls
        for await (const chunk of ollamaStream) {
          if (chunk.content) {
            accumulatedContent += chunk.content;
            fullContent.push(chunk);
          }
        }
        
        // Look for function calls in the accumulated content or infer from intent
        let match = null;
        let functionCall = null; // Define functionCall variable here
        const explicitFuncCallRegex = /<function_calls>([\s\S]*?)<\/function_calls>/;
        match = accumulatedContent.match(explicitFuncCallRegex);
        
        // If no explicit function call format found, try to detect if this is a command request
        if (!match) {
          console.log("Checking for command intent in: " + accumulatedContent.substring(0, 200));
          
          // Extract command from text based on different patterns
          let command = null;
          
          // Common command patterns
          const commandPatterns = [
            // Direct command requests
            /run\s+`([^`]+)`/i,
            /run\s+"([^"]+)"/i,
            /run\s+'([^']+)'/i,
            /run\s+the\s+command\s+`([^`]+)`/i,
            /run\s+the\s+command\s+"([^"]+)"/i,
            /run\s+the\s+command\s+'([^']+)'/i,
            /execute\s+`([^`]+)`/i,
            /execute\s+"([^"]+)"/i,
            /execute\s+'([^']+)'/i,
            /execute\s+the\s+command\s+`([^`]+)`/i,
            /execute\s+the\s+command\s+"([^"]+)"/i,
            /execute\s+the\s+command\s+'([^']+)'/i,
            
            // Common specific commands
            /run git (status|diff|log|branch|checkout|pull|push|fetch|clone|add|commit|reset|revert|stash|merge|rebase)/i,
            /execute git (status|diff|log|branch|checkout|pull|push|fetch|clone|add|commit|reset|revert|stash|merge|rebase)/i,
            /run ls/i,
            /run cd/i,
            /run mkdir/i,
            /run rm/i,
            /run cat/i,
            /run grep/i,
            /run find/i,
            /run echo/i,
            /run touch/i,
            
            // Simple run/execute commands with directly following term and no quotes
            /run ([a-zA-Z0-9_\-\.\/]+(\s+[^\n\."\']+)?)/i,
            /execute ([a-zA-Z0-9_\-\.\/]+(\s+[^\n\."\']+)?)/i,
            
            // Inferring common commands
            /you.* need to run git status/i,
            /you.* run the git status command/i,
            /I.* need to run git status/i,
            /I.* run git status for/i,
            /using the git status command/i,
            /To check .* git status/i,
            /To see .* git status/i,
            /to check .* run "([^"]+)"/i,
            /to check .* run '([^']+)'/i,
            /to check .* run `([^`]+)`/i,
            /to see .* run "([^"]+)"/i,
            /to see .* run '([^']+)'/i,
            /to see .* run `([^`]+)`/i,
            /to view .* run "([^"]+)"/i,
            /to view .* run '([^']+)'/i,
            /to view .* run `([^`]+)`/i,
            
            // Expressions like "you should run" or "you need to run"
            /you (?:should|need to|can|could|would|must) run\s+([a-zA-Z0-9_\-\.\/]+(\s+[^\n\."\'])?)/i,
            /you (?:should|need to|can|could|would|must) run\s+"([^"]+)"/i,
            /you (?:should|need to|can|could|would|must) run\s+'([^']+)'/i,
            /you (?:should|need to|can|could|would|must) run\s+`([^`]+)`/i,
            
            // Self-description of actions like "I'll run" or "I need to run"
            /I(?:'ll| will| should| need to) run\s+([a-zA-Z0-9_\-\.\/]+(\s+[^\n\."\'])?)/i,
            /I(?:'ll| will| should| need to) run\s+"([^"]+)"/i,
            /I(?:'ll| will| should| need to) run\s+'([^']+)'/i,
            /I(?:'ll| will| should| need to) run\s+`([^`]+)`/i,
          ];
          
          // Try to extract command based on patterns
          for (const pattern of commandPatterns) {
            const match = accumulatedContent.match(pattern);
            if (match) {
              // Special handling for git command patterns
              if (pattern.toString().includes('(add|status|diff|log') && match[0]) {
                // For git command patterns, extract the full command with args
                if (match[0].startsWith('run ') || match[0].startsWith('execute ')) {
                  // Remove the "run" or "execute" prefix
                  command = match[0].replace(/^(run|execute)\s+/, '');
                } else {
                  command = match[0];
                }
                console.log("Git command extracted from pattern match:", command);
                break;
              } else if (match[1]) {
                // For most patterns, use capture group 1
                command = match[1];
                console.log("Command extracted from pattern match:", command);
                break;
              }
            }
          }
          
          // If we didn't extract a command but it seems like a specific git command
          
          // Check for git status
          if (!command && (
              accumulatedContent.toLowerCase().includes("git status") ||
              (accumulatedContent.toLowerCase().includes("run git") && accumulatedContent.toLowerCase().includes("status")) ||
              (accumulatedContent.toLowerCase().includes("show") && accumulatedContent.toLowerCase().includes("status")) ||
              (accumulatedContent.toLowerCase().includes("user") && accumulatedContent.toLowerCase().includes("git") && accumulatedContent.toLowerCase().includes("status"))
          )) {
            command = "git status";
            console.log("Defaulting to git status command based on context");
          }
          
          // Check for git add
          else if (!command && (
              (accumulatedContent.toLowerCase().includes("git add") || 
              (accumulatedContent.toLowerCase().includes("run git") && accumulatedContent.toLowerCase().includes("add")) ||
              accumulatedContent.toLowerCase().includes("stage") && accumulatedContent.toLowerCase().includes("changes"))
          )) {
            // Try to extract file paths after git add
            const addMatch = accumulatedContent.match(/git add\s+([^\n]+)/i) || 
                            accumulatedContent.match(/run git add\s+([^\n]+)/i) ||
                            accumulatedContent.match(/execute git add\s+([^\n]+)/i);
            
            if (addMatch && addMatch[1]) {
              command = `git add ${addMatch[1]}`;
            } else {
              command = "git add ."; // Default to adding all changes
            }
            console.log("Detected git add command:", command);
          }
          
          // Check for git commit
          else if (!command && (
              accumulatedContent.toLowerCase().includes("git commit") ||
              (accumulatedContent.toLowerCase().includes("run git") && accumulatedContent.toLowerCase().includes("commit")) ||
              (accumulatedContent.toLowerCase().includes("execute git") && accumulatedContent.toLowerCase().includes("commit")) ||
              (accumulatedContent.toLowerCase().includes("commit") && accumulatedContent.toLowerCase().includes("changes"))
          )) {
            // Try to extract commit message using -m flag
            const commitMatch = accumulatedContent.match(/git commit\s+-m\s+["']([^"']+)["']/i) || 
                              accumulatedContent.match(/run git commit\s+-m\s+["']([^"']+)["']/i) ||
                              accumulatedContent.match(/execute git commit\s+-m\s+["']([^"']+)["']/i);
            
            // Try to extract commit message with -am flag
            const commitAllMatch = accumulatedContent.match(/git commit\s+-am\s+["']([^"']+)["']/i) || 
                                  accumulatedContent.match(/run git commit\s+-am\s+["']([^"']+)["']/i) ||
                                  accumulatedContent.match(/execute git commit\s+-am\s+["']([^"']+)["']/i);
            
            if (commitMatch && commitMatch[1]) {
              command = `git commit -m "${commitMatch[1]}"`;
            } else if (commitAllMatch && commitAllMatch[1]) {
              command = `git commit -am "${commitAllMatch[1]}"`;
            } else {
              // Default commit message
              command = `git commit -m "Changes from Claude Code session"`;
            }
            console.log("Detected git commit command:", command);
          }
          
          // Check for git push
          else if (!command && (
              accumulatedContent.toLowerCase().includes("git push") ||
              (accumulatedContent.toLowerCase().includes("run git") && accumulatedContent.toLowerCase().includes("push")) ||
              (accumulatedContent.toLowerCase().includes("execute git") && accumulatedContent.toLowerCase().includes("push")) ||
              (accumulatedContent.toLowerCase().includes("push") && accumulatedContent.toLowerCase().includes("changes") && accumulatedContent.toLowerCase().includes("remote"))
          )) {
            // Try to extract specific branch or remote info
            const pushMatch = accumulatedContent.match(/git push\s+([^\n]+)/i) || 
                            accumulatedContent.match(/run git push\s+([^\n]+)/i) ||
                            accumulatedContent.match(/execute git push\s+([^\n]+)/i);
            
            if (pushMatch && pushMatch[1]) {
              command = `git push ${pushMatch[1]}`;
            } else {
              // Default to simple push with origin tracking
              command = "git push";
            }
            console.log("Detected git push command:", command);
          }
          
          // Check for thinking content and extract commands from there
          if (!command && accumulatedContent.includes("<think>")) {
            console.log("Detected thinking content, examining for command intents");
            const thinkingMatch = accumulatedContent.match(/<think>([\s\S]*?)<\/think>/);
            if (thinkingMatch && thinkingMatch[1]) {
              const thinkingContent = thinkingMatch[1];
              
              // Detect git commands in thinking
              if (thinkingContent.includes("git status")) {
                command = "git status";
                console.log("Extracted git status command from thinking content");
              } else if (thinkingContent.includes("git diff")) {
                command = "git diff";
                console.log("Extracted git diff command from thinking content");
              } else if (thinkingContent.includes("git log")) {
                command = "git log";
                console.log("Extracted git log command from thinking content");
              } else if (thinkingContent.includes("git add")) {
                command = "git add .";
                console.log("Extracted git add command from thinking content");
              } else if (thinkingContent.includes("git commit")) {
                command = "git commit -m \"Changes from Claude Code session\"";
                console.log("Extracted git commit command from thinking content");
              } else if (thinkingContent.includes("git push")) {
                command = "git push";
                console.log("Extracted git push command from thinking content");
              }
            }
          }
          
          // If we were able to extract or determine a command, create a function call
          if (command) {
            console.log("Command intent detected, synthesizing Bash function call for:", command);
            // Create proper function call format
            const funcName = "Bash";
            const funcArgs = { command: command };
            
            // Skip the pattern matching and directly create the function call object
            match = ["<function_calls>", "<function_calls>"]; // Fake match object
            
            // Override function call detection directly
            functionCall = {
              name: funcName,
              arguments: funcArgs
            };
          }
        }
        
        // If function call is found, properly format and return it
        if (match) {
          console.log("Function call detected in Ollama output");
          
          // If we already have a function call from our intent detection, use it directly
          let functionName, params;
          
          if (functionCall) {
            console.log("Using pre-built function call:", functionCall.name);
            functionName = functionCall.name;
            params = functionCall.arguments;
          } else {
            // Extract function name the original way
            const nameMatch = match[1]?.match(/<invoke name="([^"]+)">/);
            if (!nameMatch) {
              console.error("Could not parse function name from Ollama output");
              return emitTextStream(accumulatedContent, opts.model);
            }
            
            functionName = nameMatch[1];
            
            // Extract parameters
            params = {};
            const paramMatches = [...match[1].matchAll(/<parameter name="([^"]+)">([\s\S]*?)<\/parameter>/g)];
            
            for (const paramMatch of paramMatches) {
              const [_, paramName, paramValue] = paramMatch;
              try {
                params[paramName] = JSON.parse(paramValue);
              } catch (e) {
                params[paramName] = paramValue;
              }
            }
          }
          
          // Return a function call stream
          return (async function* () {
            // Debug logging for enhanced troubleshooting
            console.log(`EXECUTING FUNCTION: ${functionName} with args:`, JSON.stringify(params));
            
            // DIRECT EXECUTION: Execute the Bash command if this is a Bash function call
            // This bypasses the normal OpenAI function calling flow
            if (functionName === "Bash" && params.command) {
              try {
                console.log(`DIRECT EXECUTION: Running Bash command: ${params.command}`);
                // Use dynamic import for ES modules
                const childProcess = await import('child_process');
                const execSync = childProcess.execSync;
                
                // Execute the command and capture output
                const cmdOutput = execSync(params.command, { 
                  encoding: 'utf8',
                  maxBuffer: 10 * 1024 * 1024, // 10MB buffer
                  timeout: 30000 // 30 second timeout
                });
                
                console.log("Command executed successfully.");
                console.log(`Command output (first 200 chars): ${cmdOutput.substring(0, 200)}`);
                
                // Return command output as a regular message instead of function call
                yield {
                  id: `ollama-cmd-${Date.now()}`,
                  object: 'chat.completion.chunk',
                  created: Math.floor(Date.now() / 1000),
                  model: opts.model,
                  choices: [{
                    index: 0,
                    delta: { 
                      role: 'assistant',
                      content: `Here's the output of \`${params.command}\`:\n\n\`\`\`\n${cmdOutput}\n\`\`\``,
                    },
                    finish_reason: null
                  }]
                };
                
                // Final message
                yield {
                  id: `ollama-cmd-done-${Date.now()}`,
                  object: 'chat.completion.chunk',
                  created: Math.floor(Date.now() / 1000),
                  model: opts.model,
                  choices: [{
                    index: 0,
                    delta: {},
                    finish_reason: 'stop'
                  }]
                };
                
                return;
              } catch (err) {
                console.error("Error executing command:", err);
                
                // Return the error as a message
                yield {
                  id: `ollama-cmd-error-${Date.now()}`,
                  object: 'chat.completion.chunk',
                  created: Math.floor(Date.now() / 1000),
                  model: opts.model,
                  choices: [{
                    index: 0,
                    delta: { 
                      role: 'assistant',
                      content: `Error executing \`${params.command}\`:\n\n\`\`\`\n${err.message || err}\n\`\`\``,
                    },
                    finish_reason: null
                  }]
                };
                
                // Final message
                yield {
                  id: `ollama-cmd-error-done-${Date.now()}`,
                  object: 'chat.completion.chunk',
                  created: Math.floor(Date.now() / 1000),
                  model: opts.model,
                  choices: [{
                    index: 0,
                    delta: {},
                    finish_reason: 'stop'
                  }]
                };
                
                return;
              }
            }
            
            // If not a direct Bash execution or execution failed, fall back to standard function call format
            // Ensure proper OpenAI format for function calls
            // 1. First yield: role + function_call with name
            const firstYield = {
              id: `ollama-fc-${Date.now()}`,
              object: 'chat.completion.chunk',
              created: Math.floor(Date.now() / 1000),
              model: opts.model,
              choices: [{
                index: 0,
                delta: { 
                  role: 'assistant',
                  content: null, // Important: set content to null for function calls
                  function_call: {
                    name: functionName
                  }
                },
                finish_reason: null
              }]
            };
            console.log("Yielding function call start:", JSON.stringify(firstYield));
            yield firstYield;
            
            // 2. Second yield: function arguments (complete)
            // Stringify with proper formatting to ensure valid JSON
            const argsString = JSON.stringify(params);
            console.log("Function args:", argsString);
            
            const secondYield = {
              id: `ollama-fc-args-${Date.now()}`,
              object: 'chat.completion.chunk',
              created: Math.floor(Date.now() / 1000),
              model: opts.model,
              choices: [{
                index: 0,
                delta: { 
                  function_call: {
                    arguments: argsString
                  },
                  content: null // Important: ensure content is null for all function call parts
                },
                finish_reason: null
              }]
            };
            console.log("Yielding function arguments:", JSON.stringify(secondYield));
            yield secondYield;
            
            // 3. Final yield: mark as done with function_call reason code
            const finalYield = {
              id: `ollama-fc-done-${Date.now()}`,
              object: 'chat.completion.chunk',
              created: Math.floor(Date.now() / 1000),
              model: opts.model,
              choices: [{
                index: 0,
                delta: {
                  content: null // Important: ensure content is consistent across all chunks
                },
                finish_reason: 'function_call'
              }]
            };
            console.log("Yielding function call completion:", JSON.stringify(finalYield));
            yield finalYield;
          })();
        }
        
        // If no function call, proceed with normal text streaming
        function emitTextStream(content, model) {
          return (async function* () {
            let isFirstChunk = true;
            
            // Generate a single chunk with the full content
            yield {
              id: `ollama-${Date.now()}`,
              object: 'chat.completion.chunk',
              created: Math.floor(Date.now() / 1000),
              model: model,
              choices: [{
                index: 0,
                delta: { 
                  role: 'assistant',
                  content: content 
                },
                finish_reason: 'stop'
              }]
            };
          })();
        }
        
        // Return original format if no function call found
        return emitTextStream(accumulatedContent, opts.model);
      } catch (error) {
        console.error("Ollama client error:", error.message);
        throw error;
      }
    }
    
    // NORMAL OPENAI PATH
    // Use different endpoint paths for Ollama vs OpenAI
    const endpoint = isOllama 
      ? `${baseURL}/api/chat` 
      : `${baseURL}/chat/completions`;
    console.log(`Making request to: ${endpoint}`);
    
    // Standard OpenAI format
    const requestBody = { ...opts, stream: true };
    
    if (process.env.VERBOSE) {
      console.log("Request body (first 200 chars):", 
        JSON.stringify(requestBody).substring(0, 200));
    }
    
    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers,
        body: JSON.stringify(requestBody),
        dispatcher: proxy,
      });
      
      if (process.env.VERBOSE) {
        console.log(`Response status: ${response.status}`);
      }
      
      if (!response.ok) {
        let errorData;
        try {
          // Clone the response first to avoid "Body already used" errors
          const errorResponse = response.clone();
          errorData = await errorResponse.json() as { error?: { message: string } };
          console.error("API error:", errorData.error?.message || JSON.stringify(errorData));
        } catch (e) {
          // If not JSON, get text
          const errorResponse = response.clone();
          const text = await errorResponse.text();
          console.error("API error text:", text);
          errorData = { error: { message: text } };
        }
        throw new Error(`API request failed: ${errorData.error?.message || JSON.stringify(errorData)}`);
      }
      
      // Clone the response to avoid "Body already used" errors
      const clonedResponse = response.clone();
      return createStreamProcessor(clonedResponse.body as any);
    } catch (error) {
      console.error("API connection error:", error.message);
      throw error;
    }
  }
  
  try {
    // Handle Ollama separately for non-streaming too
    if (isOllama) {
      console.log("Using direct Ollama client (non-streaming)");
      
      // Check if we have a custom system prompt for this model in the models config
      const modelConfig = getModelConfig(opts.model);
      const customSystemPrompt = modelConfig?.customSystemPrompt;
      
      // Create a copy of messages to potentially modify
      let modifiedMessages = [...opts.messages];
      
      // If we have a custom system prompt, replace all system messages with our custom one
      if (customSystemPrompt) {
        // Remove existing system messages
        modifiedMessages = modifiedMessages.filter(msg => msg.role !== 'system');
        
        // Add our custom system prompt at the beginning
        modifiedMessages.unshift({
          role: 'system',
          content: customSystemPrompt
        });
        
        console.log("Using custom system prompt for model (non-streaming):", opts.model);
      }
      
      // For non-streaming, we'll collect all chunks into one message
      const ollamaStream = await getOllamaCompletion(
        baseURL, 
        opts.model, 
        modifiedMessages,
        {
          temperature: opts.temperature,
          max_tokens: opts.max_tokens
        }
      );
      
      // Collect all content from stream
      let fullContent = '';
      for await (const chunk of ollamaStream) {
        if (chunk.content) {
          fullContent += chunk.content;
        }
      }
      
      // Look for function calls in the full content or infer from intent
      let match = null;
      let functionCall = null; // Define functionCall variable here
      const explicitFuncCallRegex = /<function_calls>([\s\S]*?)<\/function_calls>/;
      match = fullContent.match(explicitFuncCallRegex);
      
      // If no explicit function call format found, try to detect if this is a command request
      if (!match) {
        console.log("Checking for command intent in non-streaming: " + fullContent.substring(0, 200));
        
        // Extract command from text based on different patterns
        let command = null;
        
        // Common command patterns
        const commandPatterns = [
          // Direct command requests
          /run\s+`([^`]+)`/i,
          /run\s+"([^"]+)"/i,
          /run\s+'([^']+)'/i,
          /run\s+the\s+command\s+`([^`]+)`/i,
          /run\s+the\s+command\s+"([^"]+)"/i,
          /run\s+the\s+command\s+'([^']+)'/i,
          /execute\s+`([^`]+)`/i,
          /execute\s+"([^"]+)"/i,
          /execute\s+'([^']+)'/i,
          /execute\s+the\s+command\s+`([^`]+)`/i,
          /execute\s+the\s+command\s+"([^"]+)"/i,
          /execute\s+the\s+command\s+'([^']+)'/i,
          
          // Extract full git commands with arguments
          /run git (add|status|diff|log|branch|checkout|pull|push|fetch|clone|commit|reset|revert|stash|merge|rebase)(\s+[^\n]+)?/i,
          /execute git (add|status|diff|log|branch|checkout|pull|push|fetch|clone|commit|reset|revert|stash|merge|rebase)(\s+[^\n]+)?/i,
          /git (add|status|diff|log|branch|checkout|pull|push|fetch|clone|commit|reset|revert|stash|merge|rebase)(\s+[^\n]+)?/i,
          /run ls/i,
          /run cd/i,
          /run mkdir/i,
          /run rm/i,
          /run cat/i,
          /run grep/i,
          /run find/i,
          /run echo/i,
          /run touch/i,
          
          // Simple run/execute commands with directly following term and no quotes
          /run ([a-zA-Z0-9_\-\.\/]+(\s+[^\n\."\']+)?)/i,
          /execute ([a-zA-Z0-9_\-\.\/]+(\s+[^\n\."\']+)?)/i,
          
          // Inferring common commands
          /you.* need to run git status/i,
          /you.* run the git status command/i,
          /I.* need to run git status/i,
          /I.* run git status for/i,
          /using the git status command/i,
          /To check .* git status/i,
          /To see .* git status/i,
          /to check .* run "([^"]+)"/i,
          /to check .* run '([^']+)'/i,
          /to check .* run `([^`]+)`/i,
          /to see .* run "([^"]+)"/i,
          /to see .* run '([^']+)'/i,
          /to see .* run `([^`]+)`/i,
          /to view .* run "([^"]+)"/i,
          /to view .* run '([^']+)'/i,
          /to view .* run `([^`]+)`/i,
          
          // Expressions like "you should run" or "you need to run"
          /you (?:should|need to|can|could|would|must) run\s+([a-zA-Z0-9_\-\.\/]+(\s+[^\n\."\'])?)/i,
          /you (?:should|need to|can|could|would|must) run\s+"([^"]+)"/i,
          /you (?:should|need to|can|could|would|must) run\s+'([^']+)'/i,
          /you (?:should|need to|can|could|would|must) run\s+`([^`]+)`/i,
          
          // Self-description of actions like "I'll run" or "I need to run"
          /I(?:'ll| will| should| need to) run\s+([a-zA-Z0-9_\-\.\/]+(\s+[^\n\."\'])?)/i,
          /I(?:'ll| will| should| need to) run\s+"([^"]+)"/i,
          /I(?:'ll| will| should| need to) run\s+'([^']+)'/i,
          /I(?:'ll| will| should| need to) run\s+`([^`]+)`/i,
        ];
        
        // Try to extract command based on patterns
        for (const pattern of commandPatterns) {
          const match = fullContent.match(pattern);
          if (match) {
            // Special handling for git command patterns
            if (pattern.toString().includes('(add|status|diff|log') && match[0]) {
              // For git command patterns, extract the full command with args
              if (match[0].startsWith('run ') || match[0].startsWith('execute ')) {
                // Remove the "run" or "execute" prefix
                command = match[0].replace(/^(run|execute)\s+/, '');
              } else {
                command = match[0];
              }
              console.log("Git command extracted from pattern match (non-streaming):", command);
              break;
            } else if (match[1]) {
              // For most patterns, use capture group 1
              command = match[1];
              console.log("Command extracted from pattern match (non-streaming):", command);
              break;
            }
          }
        }
        
        // If we didn't extract a command but it seems like a specific git command
        
        // Check for git status
        if (!command && (
            fullContent.toLowerCase().includes("git status") ||
            (fullContent.toLowerCase().includes("run git") && fullContent.toLowerCase().includes("status")) ||
            (fullContent.toLowerCase().includes("show") && fullContent.toLowerCase().includes("status")) ||
            (fullContent.toLowerCase().includes("user") && fullContent.toLowerCase().includes("git") && fullContent.toLowerCase().includes("status"))
        )) {
          command = "git status";
          console.log("Defaulting to git status command based on context (non-streaming)");
        }
        
        // Check for git add
        else if (!command && (
            (fullContent.toLowerCase().includes("git add") || 
            (fullContent.toLowerCase().includes("run git") && fullContent.toLowerCase().includes("add")) ||
            fullContent.toLowerCase().includes("stage") && fullContent.toLowerCase().includes("changes"))
        )) {
          // Try to extract file paths after git add
          const addMatch = fullContent.match(/git add\s+([^\n]+)/i) || 
                          fullContent.match(/run git add\s+([^\n]+)/i) ||
                          fullContent.match(/execute git add\s+([^\n]+)/i);
          
          if (addMatch && addMatch[1]) {
            command = `git add ${addMatch[1]}`;
          } else {
            command = "git add ."; // Default to adding all changes
          }
          console.log("Detected git add command (non-streaming):", command);
        }
        
        // Check for git commit
        else if (!command && (
            fullContent.toLowerCase().includes("git commit") ||
            (fullContent.toLowerCase().includes("run git") && fullContent.toLowerCase().includes("commit")) ||
            (fullContent.toLowerCase().includes("execute git") && fullContent.toLowerCase().includes("commit")) ||
            (fullContent.toLowerCase().includes("commit") && fullContent.toLowerCase().includes("changes"))
        )) {
          // Try to extract commit message using -m flag
          const commitMatch = fullContent.match(/git commit\s+-m\s+["']([^"']+)["']/i) || 
                            fullContent.match(/run git commit\s+-m\s+["']([^"']+)["']/i) ||
                            fullContent.match(/execute git commit\s+-m\s+["']([^"']+)["']/i);
          
          // Try to extract commit message with -am flag
          const commitAllMatch = fullContent.match(/git commit\s+-am\s+["']([^"']+)["']/i) || 
                                fullContent.match(/run git commit\s+-am\s+["']([^"']+)["']/i) ||
                                fullContent.match(/execute git commit\s+-am\s+["']([^"']+)["']/i);
          
          if (commitMatch && commitMatch[1]) {
            command = `git commit -m "${commitMatch[1]}"`;
          } else if (commitAllMatch && commitAllMatch[1]) {
            command = `git commit -am "${commitAllMatch[1]}"`;
          } else {
            // Default commit message
            command = `git commit -m "Changes from Claude Code session"`;
          }
          console.log("Detected git commit command (non-streaming):", command);
        }
        
        // Check for git push
        else if (!command && (
            fullContent.toLowerCase().includes("git push") ||
            (fullContent.toLowerCase().includes("run git") && fullContent.toLowerCase().includes("push")) ||
            (fullContent.toLowerCase().includes("execute git") && fullContent.toLowerCase().includes("push")) ||
            (fullContent.toLowerCase().includes("push") && fullContent.toLowerCase().includes("changes") && fullContent.toLowerCase().includes("remote"))
        )) {
          // Try to extract specific branch or remote info
          const pushMatch = fullContent.match(/git push\s+([^\n]+)/i) || 
                          fullContent.match(/run git push\s+([^\n]+)/i) ||
                          fullContent.match(/execute git push\s+([^\n]+)/i);
          
          if (pushMatch && pushMatch[1]) {
            command = `git push ${pushMatch[1]}`;
          } else {
            // Default to simple push with origin tracking
            command = "git push";
          }
          console.log("Detected git push command (non-streaming):", command);
        }
        
        // Check for thinking content and extract commands from there
        if (!command && fullContent.includes("<think>")) {
          console.log("Detected thinking content in non-streaming, examining for command intents");
          const thinkingMatch = fullContent.match(/<think>([\s\S]*?)<\/think>/);
          if (thinkingMatch && thinkingMatch[1]) {
            const thinkingContent = thinkingMatch[1];
            
            // Detect git commands in thinking
            if (thinkingContent.includes("git status")) {
              command = "git status";
              console.log("Extracted git status command from thinking content (non-streaming)");
            } else if (thinkingContent.includes("git diff")) {
              command = "git diff";
              console.log("Extracted git diff command from thinking content (non-streaming)");
            } else if (thinkingContent.includes("git log")) {
              command = "git log"; 
              console.log("Extracted git log command from thinking content (non-streaming)");
            } else if (thinkingContent.includes("git add")) {
              command = "git add .";
              console.log("Extracted git add command from thinking content (non-streaming)");
            } else if (thinkingContent.includes("git commit")) {
              command = "git commit -m \"Changes from Claude Code session\"";
              console.log("Extracted git commit command from thinking content (non-streaming)");
            } else if (thinkingContent.includes("git push")) {
              command = "git push";
              console.log("Extracted git push command from thinking content (non-streaming)");
            }
          }
        }
        
        // If we were able to extract or determine a command, create a function call
        if (command) {
          console.log("Command intent detected in non-streaming, synthesizing Bash function call for:", command);
          // Create proper function call format
          const funcName = "Bash";
          const funcArgs = { command: command };
          
          // Skip the pattern matching and directly create the function call object
          match = ["<function_calls>", "<function_calls>"]; // Fake match object
          
          // Override function call detection directly
          functionCall = {
            name: funcName,
            arguments: funcArgs
          };
        }
      }
      
      // If function call is found, properly format and return it
      if (match) {
        console.log("Function call detected in Ollama non-streaming output");
        
        // If we already have a function call from our intent detection, use it directly
        let functionName, params;
        
        if (functionCall) {
          console.log("Using pre-built function call in non-streaming:", functionCall.name);
          functionName = functionCall.name;
          params = functionCall.arguments;
        } else {
          // Extract function name from traditional format
          const nameMatch = match[1]?.match(/<invoke name="([^"]+)">/);
          if (!nameMatch) {
            console.error("Could not parse function name from Ollama output");
            // Fall back to regular content
            return {
              id: `ollama-${Date.now()}`,
              object: 'chat.completion',
              created: Math.floor(Date.now() / 1000),
              model: opts.model,
              choices: [{
                index: 0,
                message: {
                  role: 'assistant',
                  content: fullContent
                },
                finish_reason: 'stop'
              }]
            };
          }
          
          functionName = nameMatch[1];
          
          // Extract parameters
          params = {};
          const paramMatches = [...match[1].matchAll(/<parameter name="([^"]+)">([\s\S]*?)<\/parameter>/g)];
          
          for (const paramMatch of paramMatches) {
            const [_, paramName, paramValue] = paramMatch;
            try {
              params[paramName] = JSON.parse(paramValue);
            } catch (e) {
              params[paramName] = paramValue;
            }
          }
        }
        
        // Enhanced debug logging for non-streaming function calls
        console.log(`NON-STREAMING FUNCTION EXECUTION: ${functionName} with args:`, JSON.stringify(params));
        
        // DIRECT EXECUTION: Execute the Bash command if this is a Bash function call
        // This bypasses the normal OpenAI function calling flow
        if (functionName === "Bash" && params.command) {
          try {
            console.log(`DIRECT EXECUTION (non-streaming): Running Bash command: ${params.command}`);
            // Use dynamic import for ES modules
            const childProcess = await import('child_process');
            const execSync = childProcess.execSync;
            
            // Execute the command and capture output
            const cmdOutput = execSync(params.command, { 
              encoding: 'utf8',
              maxBuffer: 10 * 1024 * 1024, // 10MB buffer
              timeout: 30000 // 30 second timeout
            });
            
            console.log("Command executed successfully (non-streaming).");
            console.log(`Command output (first 200 chars): ${cmdOutput.substring(0, 200)}`);
            
            // Return command output as a regular message instead of function call
            return {
              id: `ollama-cmd-${Date.now()}`,
              object: 'chat.completion',
              created: Math.floor(Date.now() / 1000),
              model: opts.model,
              choices: [{
                index: 0,
                message: {
                  role: 'assistant',
                  content: `Here's the output of \`${params.command}\`:\n\n\`\`\`\n${cmdOutput}\n\`\`\``,
                },
                finish_reason: 'stop'
              }]
            };
          } catch (err) {
            console.error("Error executing command (non-streaming):", err);
            
            // Return the error as a message
            return {
              id: `ollama-cmd-error-${Date.now()}`,
              object: 'chat.completion',
              created: Math.floor(Date.now() / 1000),
              model: opts.model,
              choices: [{
                index: 0,
                message: {
                  role: 'assistant',
                  content: `Error executing \`${params.command}\`:\n\n\`\`\`\n${err.message || err}\n\`\`\``,
                },
                finish_reason: 'stop'
              }]
            };
          }
        }
        
        // Format arguments properly to ensure valid JSON (fallback path)
        const argsString = JSON.stringify(params);
        console.log("Non-streaming function args:", argsString);
        
        // Return a proper function call response with explicit format for OpenAI compatibility
        const response = {
          id: `ollama-fc-${Date.now()}`,
          object: 'chat.completion',
          created: Math.floor(Date.now() / 1000),
          model: opts.model,
          choices: [{
            index: 0,
            message: {
              role: 'assistant',
              function_call: {
                name: functionName,
                arguments: argsString
              },
              // Important: content must be null for function calls according to OpenAI specs
              content: null
            },
            finish_reason: 'function_call'
          }]
        };
        
        console.log("Returning non-streaming function call:", JSON.stringify(response, null, 2).substring(0, 500));
        return response;
      }
      
      // Regular text response if no function call found
      return {
        id: `ollama-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: opts.model,
        choices: [{
          index: 0,
          message: {
            role: 'assistant',
            content: fullContent
          },
          finish_reason: 'stop'
        }]
      };
    }
    
    // Standard OpenAI client path
    // Use different endpoint paths for Ollama vs OpenAI
    const endpoint = isOllama 
      ? `${baseURL}/api/chat` 
      : `${baseURL}/chat/completions`;
    console.log(`Making non-streaming request to: ${endpoint}`);
    
    // Standard OpenAI format
    const requestBody = opts;
    
    if (process.env.VERBOSE) {
      console.log("Non-streaming request body (first 200 chars):", 
        JSON.stringify(requestBody).substring(0, 200));
    }
    
    const response = await fetch(endpoint, {
      method: 'POST',
      headers,
      body: JSON.stringify(requestBody),
      dispatcher: proxy,
    });
    
    if (!response.ok) {
      let errorData;
      try {
        // Clone the response first to avoid "Body already used" errors
        const errorResponse = response.clone();
        errorData = await errorResponse.json() as { error?: { message: string } };
      } catch (e) {
        const errorResponse = response.clone();
        const text = await errorResponse.text();
        errorData = { error: { message: text } };
      }
      throw new Error(`API request failed: ${errorData.error?.message || JSON.stringify(errorData)}`);
    }
    
    // Clone the response to avoid "Body already used" errors
    const responseForJson = response.clone();
    return responseForJson.json() as Promise<OpenAI.ChatCompletion>;
  } catch (error) {
    console.error("API request error:", error.message);
    throw error;
  }
}

export function createStreamProcessor(
  stream: any
): AsyncGenerator<OpenAI.ChatCompletionChunk, void, unknown> {
  if (!stream) {
    throw new Error("Stream is null or undefined")
  }
  
  return (async function* () {
    const reader = stream.getReader()
    const decoder = new TextDecoder('utf-8')
    let buffer = ''
    
    // Check if this is an Ollama stream based on URL or provider
    const config = getGlobalConfig();
    const isOllama = (config.largeModelBaseURL?.includes("ollama")) || 
                    (config.largeModelProvider === "ollama");
    
    // We need to track the first chunk for Ollama format
    let isFirstChunk = true;
    
    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true })
        buffer += chunk
        
        if (isOllama) {
          // Process Ollama format - newline delimited JSON
          let lineEnd = buffer.indexOf('\n')
          while (lineEnd !== -1) {
            const line = buffer.substring(0, lineEnd).trim()
            buffer = buffer.substring(lineEnd + 1)
            
            if (!line) continue;
            
            try {
              // Add debug output for raw line
              console.log("OLLAMA RAW LINE:", line);
              
              const ollamaResponse = JSON.parse(line);
              console.log("OLLAMA PARSED:", JSON.stringify(ollamaResponse));
              
              if (ollamaResponse.message && ollamaResponse.message.content) {
                // Extract the content
                const contentText = ollamaResponse.message.content;
                
                // CRITICAL DEBUG: Just log the raw content to console
                console.log(`OLLAMA CONTENT: "${contentText}"`);
                
                // Create an OpenAI-compatible chunk
                const openaiChunk: OpenAI.ChatCompletionChunk = {
                  id: `ollama-${Date.now()}`,
                  object: 'chat.completion.chunk',
                  created: Math.floor(Date.now() / 1000),
                  model: ollamaResponse.model || 'ollama-model',
                  choices: [{
                    index: 0,
                    delta: isFirstChunk 
                      ? { role: 'assistant', content: contentText }
                      : { content: contentText },
                    finish_reason: ollamaResponse.done ? 'stop' : null
                  }]
                };
                
                // Only the first chunk gets role: assistant
                if (isFirstChunk) {
                  isFirstChunk = false;
                }
                
                yield openaiChunk;
              }
            } catch (e) {
              // Skip invalid JSON - only log in verbose mode
              if (process.env.VERBOSE) {
                console.error('Error parsing Ollama JSON:', e.message);
              }
            }
            
            lineEnd = buffer.indexOf('\n')
          }
        } else {
          // Standard OpenAI format - SSE with data: prefix
          let lineEnd = buffer.indexOf('\n')
          while (lineEnd !== -1) {
            const line = buffer.substring(0, lineEnd).trim()
            buffer = buffer.substring(lineEnd + 1)
            
            if (line === 'data: [DONE]') {
              continue
            }
            
            if (line.startsWith('data: ')) {
              const data = line.slice(6).trim()
              if (!data) continue
              
              try {
                const parsed = JSON.parse(data) as OpenAI.ChatCompletionChunk
                yield parsed
              } catch (e) {
                // Skip invalid JSON - only log in verbose mode
                if (process.env.VERBOSE) {
                  console.error('Error parsing OpenAI JSON:', e.message);
                }
              }
            }
            
            lineEnd = buffer.indexOf('\n')
          }
        }
      }
      
      // Process any remaining data in the buffer
      if (buffer.trim()) {
        const lines = buffer.trim().split('\n')
        for (const line of lines) {
          if (isOllama) {
            if (line) {
              try {
                const ollamaResponse = JSON.parse(line);
                
                if (ollamaResponse.message && ollamaResponse.message.content) {
                  // Process content the same way as streaming
                  const contentText = ollamaResponse.message.content;
                  
                  const openaiChunk: OpenAI.ChatCompletionChunk = {
                    id: `ollama-${Date.now()}`,
                    object: 'chat.completion.chunk',
                    created: Math.floor(Date.now() / 1000),
                    model: ollamaResponse.model || 'ollama-model',
                    choices: [{
                      index: 0,
                      delta: isFirstChunk 
                        ? { role: 'assistant', content: contentText }
                        : { content: contentText },
                      finish_reason: ollamaResponse.done ? 'stop' : null
                    }]
                  };
                  
                  // Update after processing
                  if (isFirstChunk) {
                    isFirstChunk = false;
                  }
                  
                  yield openaiChunk;
                }
              } catch (e) {
                // Skip invalid JSON - only log in verbose mode
                if (process.env.VERBOSE) {
                  console.error('Error parsing final Ollama JSON:', e.message);
                }
              }
            }
          } else if (line.startsWith('data: ') && line !== 'data: [DONE]') {
            const data = line.slice(6).trim()
            if (!data) continue
            
            try {
              const parsed = JSON.parse(data) as OpenAI.ChatCompletionChunk
              yield parsed
            } catch (e) {
              // Skip invalid JSON - only log in verbose mode
              if (process.env.VERBOSE) {
                console.error('Error parsing final OpenAI JSON:', e.message);
              }
            }
          }
        }
      }
    } finally {
      reader.releaseLock()
    }
  })()
}

export function streamCompletion(
  stream: any
): AsyncGenerator<OpenAI.ChatCompletionChunk, void, unknown> {
  return createStreamProcessor(stream)
}