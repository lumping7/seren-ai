import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useToast } from "@/hooks/use-toast";
import { ArrowLeftIcon, BookOpenIcon, ServerIcon, CpuIcon } from "lucide-react";
import { Link } from "wouter";

export default function SetupGuide() {
  const { toast } = useToast();

  // Function to copy text to clipboard
  const copyToClipboard = (text: string, description: string) => {
    navigator.clipboard.writeText(text).then(
      () => {
        toast({
          title: "Copied to clipboard",
          description: description,
        });
      },
      () => {
        toast({
          variant: "destructive",
          title: "Copy failed",
          description: "Could not copy to clipboard",
        });
      }
    );
  };

  return (
    <div className="container mx-auto py-8 px-4 max-w-7xl">
      <div className="flex items-center mb-6 space-x-2">
        <Link href="/">
          <Button variant="outline" size="icon" className="mr-2">
            <ArrowLeftIcon className="h-4 w-4" />
          </Button>
        </Link>
        <h1 className="text-3xl font-bold flex items-center">
          <BookOpenIcon className="mr-2 h-8 w-8" />
          Seren AI Setup Guide
        </h1>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>LLM Setup Instructions</CardTitle>
            <CardDescription>
              Follow these steps to set up real AI models with Ollama
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="linux">
              <TabsList className="mb-4">
                <TabsTrigger value="linux">Linux</TabsTrigger>
                <TabsTrigger value="macos">macOS</TabsTrigger>
                <TabsTrigger value="windows">Windows</TabsTrigger>
              </TabsList>

              <TabsContent value="linux">
                <div className="space-y-4">
                  <div className="border rounded-md p-4 bg-muted/50">
                    <h3 className="font-medium mb-2">1. Install Ollama</h3>
                    <pre className="bg-black text-white p-3 rounded text-sm overflow-x-auto">
                      <code>curl -fsSL https://ollama.com/install.sh | sh</code>
                    </pre>
                    <Button 
                      variant="outline" 
                      size="sm" 
                      className="mt-2"
                      onClick={() => copyToClipboard(
                        "curl -fsSL https://ollama.com/install.sh | sh",
                        "Install command copied"
                      )}
                    >
                      Copy
                    </Button>
                  </div>

                  <div className="border rounded-md p-4 bg-muted/50">
                    <h3 className="font-medium mb-2">2. Start Ollama Service</h3>
                    <pre className="bg-black text-white p-3 rounded text-sm overflow-x-auto">
                      <code>ollama serve</code>
                    </pre>
                    <p className="text-sm mt-2 text-muted-foreground">
                      Run this in a separate terminal or as a background service
                    </p>
                  </div>

                  <div className="border rounded-md p-4 bg-muted/50">
                    <h3 className="font-medium mb-2">3. Pull Required Models</h3>
                    <pre className="bg-black text-white p-3 rounded text-sm overflow-x-auto">
                      <code># Install Qwen model (for planning and architecture)<br/>
ollama pull qwen2:7b<br/><br/>
# Install CodeLlama (for coding and implementation)<br/>
ollama pull codellama:7b</code>
                    </pre>
                    <Button 
                      variant="outline" 
                      size="sm" 
                      className="mt-2 mr-2"
                      onClick={() => copyToClipboard(
                        "ollama pull qwen2:7b",
                        "Qwen model command copied"
                      )}
                    >
                      Copy Qwen command
                    </Button>
                    <Button 
                      variant="outline" 
                      size="sm" 
                      className="mt-2"
                      onClick={() => copyToClipboard(
                        "ollama pull codellama:7b",
                        "CodeLlama model command copied"
                      )}
                    >
                      Copy CodeLlama command
                    </Button>
                  </div>

                  <div className="border rounded-md p-4 bg-muted/50">
                    <h3 className="font-medium mb-2">4. Verify Installation</h3>
                    <pre className="bg-black text-white p-3 rounded text-sm overflow-x-auto">
                      <code>ollama list</code>
                    </pre>
                    <p className="text-sm mt-2 text-muted-foreground">
                      You should see both qwen2:7b and codellama:7b in the list
                    </p>
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="macos">
                <div className="space-y-4">
                  <div className="border rounded-md p-4 bg-muted/50">
                    <h3 className="font-medium mb-2">1. Download and Install Ollama</h3>
                    <p>Download Ollama from the <a href="https://ollama.com/download/mac" className="text-primary underline" target="_blank" rel="noopener noreferrer">official website</a></p>
                    <p className="text-sm mt-2 text-muted-foreground">
                      Open the downloaded file and follow the installation instructions
                    </p>
                  </div>

                  <div className="border rounded-md p-4 bg-muted/50">
                    <h3 className="font-medium mb-2">2. Run Ollama</h3>
                    <p>Open the Ollama application from your Applications folder</p>
                    <p className="text-sm mt-2 text-muted-foreground">
                      Ollama will run in the background and be accessible via the menu bar icon
                    </p>
                  </div>

                  <div className="border rounded-md p-4 bg-muted/50">
                    <h3 className="font-medium mb-2">3. Pull Required Models</h3>
                    <p>Open Terminal and run:</p>
                    <pre className="bg-black text-white p-3 rounded text-sm overflow-x-auto">
                      <code># Install Qwen model (for planning and architecture)<br/>
ollama pull qwen2:7b<br/><br/>
# Install CodeLlama (for coding and implementation)<br/>
ollama pull codellama:7b</code>
                    </pre>
                    <Button 
                      variant="outline" 
                      size="sm" 
                      className="mt-2 mr-2"
                      onClick={() => copyToClipboard(
                        "ollama pull qwen2:7b",
                        "Qwen model command copied"
                      )}
                    >
                      Copy Qwen command
                    </Button>
                    <Button 
                      variant="outline" 
                      size="sm" 
                      className="mt-2"
                      onClick={() => copyToClipboard(
                        "ollama pull codellama:7b",
                        "CodeLlama model command copied"
                      )}
                    >
                      Copy CodeLlama command
                    </Button>
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="windows">
                <div className="space-y-4">
                  <div className="border rounded-md p-4 bg-muted/50">
                    <h3 className="font-medium mb-2">1. Download and Install Ollama</h3>
                    <p>Download Ollama from the <a href="https://ollama.com/download/windows" className="text-primary underline" target="_blank" rel="noopener noreferrer">official website</a></p>
                    <p className="text-sm mt-2 text-muted-foreground">
                      Run the installer and follow the installation instructions
                    </p>
                  </div>

                  <div className="border rounded-md p-4 bg-muted/50">
                    <h3 className="font-medium mb-2">2. Run Ollama</h3>
                    <p>Ollama should start automatically after installation</p>
                    <p className="text-sm mt-2 text-muted-foreground">
                      You can check its status in the system tray
                    </p>
                  </div>

                  <div className="border rounded-md p-4 bg-muted/50">
                    <h3 className="font-medium mb-2">3. Pull Required Models</h3>
                    <p>Open Command Prompt or PowerShell and run:</p>
                    <pre className="bg-black text-white p-3 rounded text-sm overflow-x-auto">
                      <code># Install Qwen model (for planning and architecture)<br/>
ollama pull qwen2:7b<br/><br/>
# Install CodeLlama (for coding and implementation)<br/>
ollama pull codellama:7b</code>
                    </pre>
                    <Button 
                      variant="outline" 
                      size="sm" 
                      className="mt-2 mr-2"
                      onClick={() => copyToClipboard(
                        "ollama pull qwen2:7b",
                        "Qwen model command copied"
                      )}
                    >
                      Copy Qwen command
                    </Button>
                    <Button 
                      variant="outline" 
                      size="sm" 
                      className="mt-2"
                      onClick={() => copyToClipboard(
                        "ollama pull codellama:7b",
                        "CodeLlama model command copied"
                      )}
                    >
                      Copy CodeLlama command
                    </Button>
                  </div>
                </div>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>

        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <ServerIcon className="mr-2 h-5 w-5" />
                System Requirements
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2">
                <li className="flex justify-between">
                  <span className="font-medium">RAM</span>
                  <span>16GB+ recommended</span>
                </li>
                <li className="flex justify-between">
                  <span className="font-medium">CPU</span>
                  <span>Modern x86_64 with AVX2</span>
                </li>
                <li className="flex justify-between">
                  <span className="font-medium">Disk Space</span>
                  <span>20GB+ free space</span>
                </li>
                <li className="flex justify-between">
                  <span className="font-medium">GPU</span>
                  <span>Optional but recommended</span>
                </li>
                <li className="flex justify-between">
                  <span className="font-medium">OS</span>
                  <span>Linux, macOS, or Windows</span>
                </li>
              </ul>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <CpuIcon className="mr-2 h-5 w-5" />
                Performance Tips
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div>
                <h3 className="font-medium">For Better Speed</h3>
                <p className="text-sm text-muted-foreground">Use a machine with a CUDA-compatible NVIDIA GPU for significantly faster model inference.</p>
              </div>
              <div>
                <h3 className="font-medium">Memory Management</h3>
                <p className="text-sm text-muted-foreground">Close memory-intensive applications before running the models to ensure enough RAM is available.</p>
              </div>
              <div>
                <h3 className="font-medium">Smaller Models</h3>
                <p className="text-sm text-muted-foreground">If your hardware is limited, you can use smaller models like llama3:8b which require less resources.</p>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Troubleshooting</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <div>
                <h3 className="font-medium">Connection Refused</h3>
                <p className="text-sm text-muted-foreground">Make sure Ollama is running with <code className="text-xs bg-muted p-1 rounded">ollama serve</code></p>
              </div>
              <div>
                <h3 className="font-medium">Out of Memory</h3>
                <p className="text-sm text-muted-foreground">Try freeing up RAM or using a smaller model</p>
              </div>
              <div>
                <h3 className="font-medium">Slow Performance</h3>
                <p className="text-sm text-muted-foreground">Consider using a GPU or reducing model parameters</p>
              </div>
              <div>
                <h3 className="font-medium">Need More Help?</h3>
                <p className="text-sm text-muted-foreground">Check the <a href="https://github.com/ollama/ollama/blob/main/docs/troubleshooting.md" className="text-primary underline" target="_blank" rel="noopener noreferrer">Ollama troubleshooting guide</a></p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}