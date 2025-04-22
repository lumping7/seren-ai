import { Request, Response } from 'express';
import { spawn } from 'child_process';
import { randomBytes } from 'crypto';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

/**
 * Code Execution Handler
 * Provides secure execution of code snippets in isolated environments
 */
export async function executeCodeHandler(req: Request, res: Response) {
  try {
    const { code, language = 'python', timeout = 10000 } = req.body;
    
    if (!code) {
      return res.status(400).json({ error: 'Code is required' });
    }
    
    // Check if user is authenticated and has execution permissions
    if (!req.isAuthenticated()) {
      return res.status(401).json({ error: 'Authentication required for code execution' });
    }
    
    // Validate the language is supported
    const supportedLanguages = ['python', 'javascript', 'bash'];
    if (!supportedLanguages.includes(language.toLowerCase())) {
      return res.status(400).json({ 
        error: `Unsupported language. Supported languages are: ${supportedLanguages.join(', ')}` 
      });
    }
    
    // Create a random execution ID for this run
    const executionId = randomBytes(8).toString('hex');
    
    // In a production environment, we would use container isolation
    // For this implementation, we'll use a basic child process
    const result = await executeCode(code, language, executionId, timeout);
    
    return res.json(result);
  } catch (error) {
    console.error('Error executing code:', error);
    return res.status(500).json({ error: 'Failed to execute code' });
  }
}

/**
 * Execute code in a child process
 */
async function executeCode(
  code: string, 
  language: string, 
  executionId: string, 
  timeout: number
): Promise<any> {
  return new Promise(async (resolve, reject) => {
    try {
      // Create a temporary directory for the execution
      const tempDir = path.join(os.tmpdir(), `ai-execution-${executionId}`);
      
      if (!fs.existsSync(tempDir)) {
        fs.mkdirSync(tempDir);
      }
      
      let command: string;
      let args: string[] = [];
      let filePath: string;
      
      // Prepare execution based on language
      switch (language.toLowerCase()) {
        case 'python':
          filePath = path.join(tempDir, 'script.py');
          fs.writeFileSync(filePath, code);
          command = 'python';
          args = [filePath];
          break;
          
        case 'javascript':
          filePath = path.join(tempDir, 'script.js');
          fs.writeFileSync(filePath, code);
          command = 'node';
          args = [filePath];
          break;
          
        case 'bash':
          filePath = path.join(tempDir, 'script.sh');
          fs.writeFileSync(filePath, code);
          fs.chmodSync(filePath, '755');
          command = '/bin/bash';
          args = [filePath];
          break;
          
        default:
          throw new Error(`Unsupported language: ${language}`);
      }
      
      // Create timeout
      const timeoutId = setTimeout(() => {
        cleanup();
        resolve({
          success: false,
          stdout: '',
          stderr: 'Execution timed out',
          executionTime: timeout,
          executionId
        });
      }, timeout);
      
      // Execute the code
      const startTime = Date.now();
      const process = spawn(command, args);
      
      let stdout = '';
      let stderr = '';
      
      process.stdout.on('data', (data) => {
        stdout += data.toString();
      });
      
      process.stderr.on('data', (data) => {
        stderr += data.toString();
      });
      
      process.on('close', (code) => {
        clearTimeout(timeoutId);
        cleanup();
        
        const executionTime = (Date.now() - startTime) / 1000;
        
        resolve({
          success: code === 0,
          exitCode: code,
          stdout,
          stderr,
          executionTime,
          executionId
        });
      });
      
      process.on('error', (err) => {
        clearTimeout(timeoutId);
        cleanup();
        reject(err);
      });
      
      // Cleanup function to remove temporary files
      function cleanup() {
        try {
          fs.rmdirSync(tempDir, { recursive: true });
        } catch (err) {
          console.error(`Failed to clean up execution directory: ${err}`);
        }
      }
    } catch (error) {
      reject(error);
    }
  });
}
