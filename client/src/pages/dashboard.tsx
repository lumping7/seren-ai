import React, { useState } from 'react';
import { useAuth } from '@/hooks/use-auth';
import { useToast } from '@/hooks/use-toast';
import { useQuery } from '@tanstack/react-query';
import { apiRequest } from '@/lib/queryClient';
import { 
  Loader2, 
  Code, 
  Cpu, 
  GitBranch, 
  Play, 
  Pause, 
  RefreshCw, 
  FileText, 
  ArrowRight, 
  Brain,
  Settings,
  BarChart3,
  Search,
  MessageSquare
} from 'lucide-react';

import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Textarea } from '@/components/ui/textarea';
import { Input } from '@/components/ui/input';

export default function Dashboard() {
  const { user } = useAuth();
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState('projects');
  const [selectedProject, setSelectedProject] = useState<string | null>(null);
  const [selectedOpenManusProject, setSelectedOpenManusProject] = useState<string | null>(null);
  
  // Get models status
  const { data: modelStatus, isLoading: modelsLoading } = useQuery({
    queryKey: ['/api/ai/models/status'],
    refetchInterval: 10000,
  });

  // Get continuous execution projects
  const { 
    data: projects, 
    isLoading: projectsLoading, 
    refetch: refetchProjects 
  } = useQuery({
    queryKey: ['/api/ai/continuous/project'],
    refetchInterval: selectedProject ? 5000 : false,
  });
  
  // Get OpenManus projects
  const {
    data: openManusProjects,
    isLoading: openManusProjectsLoading,
    refetch: refetchOpenManusProjects
  } = useQuery({
    queryKey: ['/api/ai/openmanus/projects'],
    refetchInterval: selectedOpenManusProject ? 5000 : false,
  });

  // Get selected project details
  const { 
    data: projectDetails, 
    isLoading: projectDetailsLoading, 
    refetch: refetchProjectDetails 
  } = useQuery({
    queryKey: ['/api/ai/continuous/project', selectedProject],
    enabled: !!selectedProject,
    refetchInterval: 3000,
  });
  
  // Get OpenManus project details
  const {
    data: openManusProjectDetails,
    isLoading: openManusProjectDetailsLoading,
    refetch: refetchOpenManusProjectDetails
  } = useQuery({
    queryKey: ['/api/ai/openmanus/project', selectedOpenManusProject],
    enabled: !!selectedOpenManusProject,
    refetchInterval: 3000,
  });

  // Create new project mutation
  const createProjectMutation = useMutation({
    mutationFn: async (data: z.infer<typeof projectSchema>) => {
      const response = await apiRequest('POST', '/api/ai/continuous/project', data);
      return await response.json();
    },
    onSuccess: () => {
      toast({
        title: 'Project created',
        description: 'Your autonomous project has been started successfully',
      });
      refetchProjects();
      projectForm.reset();
    },
    onError: (error: any) => {
      toast({
        title: 'Error creating project',
        description: error.message || 'Failed to create project',
        variant: 'destructive',
      });
    },
  });
  
  // Create new OpenManus project mutation
  const createOpenManusProjectMutation = useMutation({
    mutationFn: async (data: z.infer<typeof projectSchema>) => {
      const response = await apiRequest('POST', '/api/ai/openmanus/project', {
        name: data.name,
        description: data.description,
        language: data.language,
        framework: data.framework,
        features: [],
        constraints: []
      });
      return await response.json();
    },
    onSuccess: (data) => {
      toast({
        title: 'OpenManus Project created',
        description: 'Your agentic project has been started with project ID: ' + data.projectId,
      });
      refetchOpenManusProjects();
      projectForm.reset();
    },
    onError: (error: any) => {
      toast({
        title: 'Error creating OpenManus project',
        description: error.message || 'Failed to create project',
        variant: 'destructive',
      });
    },
  });

  // Project control mutation
  const projectControlMutation = useMutation({
    mutationFn: async ({ id, action }: { id: string, action: 'pause' | 'resume' | 'cancel' }) => {
      const response = await apiRequest('POST', `/api/ai/continuous/project/${id}/action`, { action });
      return await response.json();
    },
    onSuccess: () => {
      refetchProjectDetails();
      refetchProjects();
    },
    onError: (error: any) => {
      toast({
        title: 'Error controlling project',
        description: error.message || 'Failed to execute action',
        variant: 'destructive',
      });
    },
  });

  // Generate code mutation
  const generateCodeMutation = useMutation({
    mutationFn: async (data: z.infer<typeof codeGenSchema>) => {
      const response = await apiRequest('POST', '/api/ai/models/generate', data);
      return await response.json();
    },
    onSuccess: (data) => {
      toast({
        title: 'Code generated',
        description: 'The AI dev team has generated code based on your requirements',
      });
      
      // Update the code in the form
      if (data.code) {
        enhanceForm.setValue('code', data.code);
      }
    },
    onError: (error: any) => {
      toast({
        title: 'Error generating code',
        description: error.message || 'Failed to generate code',
        variant: 'destructive',
      });
    },
  });

  // Enhance code mutation
  const enhanceCodeMutation = useMutation({
    mutationFn: async (data: z.infer<typeof enhanceSchema>) => {
      const response = await apiRequest('POST', '/api/ai/models/enhance', data);
      return await response.json();
    },
    onSuccess: (data) => {
      toast({
        title: 'Code enhanced',
        description: `The AI dev team has ${enhanceForm.getValues('enhancement')}d your code`,
      });
      
      // Update the code in the form with the enhanced version
      if (data.enhanced_code) {
        enhanceForm.setValue('code', data.enhanced_code);
      }
    },
    onError: (error: any) => {
      toast({
        title: 'Error enhancing code',
        description: error.message || 'Failed to enhance code',
        variant: 'destructive',
      });
    },
  });

  // Project creation form
  const projectForm = useForm<z.infer<typeof projectSchema>>({
    resolver: zodResolver(projectSchema),
    defaultValues: {
      name: '',
      description: '',
      requirements: '',
      primaryModel: 'hybrid',
      language: '',
      framework: '',
      priorityLevel: 'medium',
    },
  });

  // Code generation form
  const codeGenForm = useForm<z.infer<typeof codeGenSchema>>({
    resolver: zodResolver(codeGenSchema),
    defaultValues: {
      requirements: '',
      language: '',
      framework: '',
      primaryModel: 'hybrid',
    },
  });

  // Code enhancement form
  const enhanceForm = useForm<z.infer<typeof enhanceSchema>>({
    resolver: zodResolver(enhanceSchema),
    defaultValues: {
      code: '',
      enhancement: 'optimize',
      language: '',
      requirements: '',
    },
  });

  // Handle project creation
  const onProjectSubmit = (data: z.infer<typeof projectSchema>) => {
    // Check if the advanced agentic system checkbox is checked
    const useOpenManus = data.primaryModel === 'openmanus';
    
    if (useOpenManus) {
      createOpenManusProjectMutation.mutate(data);
    } else {
      createProjectMutation.mutate(data);
    }
  };

  // Handle code generation
  const onCodeGenSubmit = (data: z.infer<typeof codeGenSchema>) => {
    generateCodeMutation.mutate(data);
  };

  // Handle code enhancement
  const onEnhanceSubmit = (data: z.infer<typeof enhanceSchema>) => {
    enhanceCodeMutation.mutate(data);
  };

  // Get status badge color based on project status
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'initializing':
        return 'bg-yellow-500';
      case 'in_progress':
        return 'bg-blue-500';
      case 'paused':
        return 'bg-orange-500';
      case 'completed':
        return 'bg-green-500';
      case 'failed':
        return 'bg-red-500';
      default:
        return 'bg-gray-500';
    }
  };

  // Get phase badge color based on project phase
  const getPhaseColor = (phase: string) => {
    switch (phase) {
      case 'requirements_analysis':
        return 'bg-purple-500';
      case 'architecture_design':
        return 'bg-indigo-500';
      case 'implementation':
        return 'bg-blue-500';
      case 'testing':
        return 'bg-cyan-500';
      case 'optimization':
        return 'bg-teal-500';
      case 'documentation':
        return 'bg-green-500';
      case 'delivery':
        return 'bg-emerald-500';
      default:
        return 'bg-gray-500';
    }
  };

  // Format timestamp to readable format
  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  if (!user) {
    return (
      <div className="h-screen flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-border" />
      </div>
    );
  }

  return (
    <div className="container mx-auto py-10">
      <h1 className="text-4xl font-bold mb-4">Seren AI Developer</h1>
      <p className="text-xl text-muted-foreground mb-8">
        Autonomous AI dev team with Qwen2.5-7b-omni and OlympicCoder-7B
      </p>

      <Tabs defaultValue="projects" value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="mb-8">
          <TabsTrigger value="projects">
            <GitBranch className="mr-2 h-4 w-4" />
            Autonomous Projects
          </TabsTrigger>
          <TabsTrigger value="direct">
            <Code className="mr-2 h-4 w-4" />
            Direct AI Coding
          </TabsTrigger>
          <TabsTrigger value="status">
            <Cpu className="mr-2 h-4 w-4" />
            System Status
          </TabsTrigger>
        </TabsList>

        {/* Autonomous Projects Tab */}
        <TabsContent value="projects" className="space-y-8">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <Card>
              <CardHeader>
                <CardTitle>Create New Project</CardTitle>
                <CardDescription>
                  Start a new autonomous software project
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Form {...projectForm}>
                  <form onSubmit={projectForm.handleSubmit(onProjectSubmit)} className="space-y-4">
                    <FormField
                      control={projectForm.control}
                      name="name"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Project Name</FormLabel>
                          <FormControl>
                            <Input placeholder="My Awesome Project" {...field} />
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                    
                    <FormField
                      control={projectForm.control}
                      name="description"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Description</FormLabel>
                          <FormControl>
                            <Input placeholder="A short description of your project" {...field} />
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />

                    <FormField
                      control={projectForm.control}
                      name="requirements"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Requirements</FormLabel>
                          <FormControl>
                            <Textarea 
                              placeholder="Detailed requirements for your project" 
                              className="min-h-[150px]" 
                              {...field} 
                            />
                          </FormControl>
                          <FormDescription>
                            Be as detailed as possible about what you want to build
                          </FormDescription>
                          <FormMessage />
                        </FormItem>
                      )}
                    />

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <FormField
                        control={projectForm.control}
                        name="language"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Language</FormLabel>
                            <FormControl>
                              <Input placeholder="e.g. JavaScript, Python" {...field} />
                            </FormControl>
                          </FormItem>
                        )}
                      />

                      <FormField
                        control={projectForm.control}
                        name="framework"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Framework</FormLabel>
                            <FormControl>
                              <Input placeholder="e.g. React, Django" {...field} />
                            </FormControl>
                          </FormItem>
                        )}
                      />
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <FormField
                        control={projectForm.control}
                        name="primaryModel"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Primary Model</FormLabel>
                            <Select 
                              onValueChange={field.onChange} 
                              defaultValue={field.value}
                            >
                              <FormControl>
                                <SelectTrigger>
                                  <SelectValue placeholder="Select a model" />
                                </SelectTrigger>
                              </FormControl>
                              <SelectContent>
                                <SelectItem value="hybrid">Hybrid (Both Models)</SelectItem>
                                <SelectItem value="qwen2.5-7b-omni">Qwen2.5-7b-omni</SelectItem>
                                <SelectItem value="olympiccoder-7b">OlympicCoder-7B</SelectItem>
                                <SelectItem value="openmanus">OpenManus (Advanced Agentic System)</SelectItem>
                              </SelectContent>
                            </Select>
                          </FormItem>
                        )}
                      />

                      <FormField
                        control={projectForm.control}
                        name="priorityLevel"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Priority</FormLabel>
                            <Select 
                              onValueChange={field.onChange} 
                              defaultValue={field.value}
                            >
                              <FormControl>
                                <SelectTrigger>
                                  <SelectValue placeholder="Select priority" />
                                </SelectTrigger>
                              </FormControl>
                              <SelectContent>
                                <SelectItem value="low">Low</SelectItem>
                                <SelectItem value="medium">Medium</SelectItem>
                                <SelectItem value="high">High</SelectItem>
                                <SelectItem value="critical">Critical</SelectItem>
                              </SelectContent>
                            </Select>
                          </FormItem>
                        )}
                      />
                    </div>

                    <Button 
                      type="submit" 
                      className="w-full"
                      disabled={createProjectMutation.isPending}
                    >
                      {createProjectMutation.isPending ? (
                        <>
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          Creating Project...
                        </>
                      ) : (
                        'Start Autonomous Project'
                      )}
                    </Button>
                  </form>
                </Form>
              </CardContent>
            </Card>

            <Tabs defaultValue="continuous" className="w-full">
              <TabsList className="w-full">
                <TabsTrigger value="continuous" className="flex-1">Continuous Execution</TabsTrigger>
                <TabsTrigger value="openmanus" className="flex-1">OpenManus Agentic</TabsTrigger>
              </TabsList>
              <TabsContent value="continuous">
                <Card>
                  <CardHeader>
                    <CardTitle>Your Projects</CardTitle>
                    <CardDescription>
                      Manage your autonomous software projects
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {projectsLoading ? (
                      <div className="flex items-center justify-center p-8">
                        <Loader2 className="h-8 w-8 animate-spin text-border" />
                      </div>
                    ) : projects && projects.projects && projects.projects.length > 0 ? (
                      <div className="space-y-4">
                        {projects.projects.map((project: any) => (
                          <Card 
                            key={project.id} 
                            className={`border-l-4 ${
                              selectedProject === project.id ? 'border-l-primary' : 'border-l-border'
                            } cursor-pointer hover:shadow-md transition-all`}
                            onClick={() => setSelectedProject(project.id)}
                          >
                            <CardContent className="p-4">
                              <div className="flex justify-between items-start">
                                <div>
                                  <h3 className="font-bold">{project.name}</h3>
                                  <p className="text-sm text-muted-foreground">{project.description}</p>
                                </div>
                                <div className="flex gap-2">
                                  <Badge className={getStatusColor(project.status)}>
                                    {project.status.replace('_', ' ')}
                                  </Badge>
                                  <Badge className={getPhaseColor(project.currentPhase)}>
                                    {project.currentPhase.replace('_', ' ')}
                                  </Badge>
                                </div>
                              </div>
                              <div className="mt-2">
                                <Progress value={project.progress || 0} className="h-2" />
                              </div>
                              <div className="text-xs text-muted-foreground mt-2">
                                Started: {formatTime(project.startTime)}
                              </div>
                            </CardContent>
                          </Card>
                        ))}
                      </div>
                    ) : (
                      <div className="text-center p-8 border rounded-lg border-dashed">
                        <p>No projects found. Create your first autonomous project!</p>
                      </div>
                    )}
                  </CardContent>
                  <CardFooter className="justify-between">
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={() => refetchProjects()}
                    >
                      <RefreshCw className="h-4 w-4 mr-2" />
                      Refresh
                    </Button>
                    {selectedProject && (
                      <div className="space-x-2">
                        <Button
                          variant="outline"
                          size="sm"
                          disabled={
                            projectControlMutation.isPending || 
                            projectDetailsLoading ||
                            (projectDetails?.status !== 'in_progress' && projectDetails?.status !== 'paused')
                          }
                          onClick={() => {
                            if (projectDetails?.status === 'in_progress') {
                              projectControlMutation.mutate({ 
                                id: selectedProject, 
                                action: 'pause' 
                              });
                            } else if (projectDetails?.status === 'paused') {
                              projectControlMutation.mutate({ 
                                id: selectedProject, 
                                action: 'resume' 
                              });
                            }
                          }}
                        >
                          {projectControlMutation.isPending ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : projectDetails?.status === 'in_progress' ? (
                            <Pause className="h-4 w-4" />
                          ) : (
                            <Play className="h-4 w-4" />
                          )}
                        </Button>
                        <Button
                          variant="destructive"
                          size="sm"
                          disabled={
                            projectControlMutation.isPending || 
                            projectDetailsLoading ||
                            projectDetails?.status === 'completed' || 
                            projectDetails?.status === 'failed'
                          }
                          onClick={() => {
                            projectControlMutation.mutate({ 
                              id: selectedProject, 
                              action: 'cancel' 
                            });
                          }}
                        >
                          Cancel
                        </Button>
                      </div>
                    )}
                  </CardFooter>
                </Card>
              </TabsContent>
              
              <TabsContent value="openmanus">
                <Card>
                  <CardHeader>
                    <CardTitle>OpenManus Agentic Projects</CardTitle>
                    <CardDescription>
                      Manage your advanced agentic system projects
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {openManusProjectsLoading ? (
                      <div className="flex items-center justify-center p-8">
                        <Loader2 className="h-8 w-8 animate-spin text-border" />
                      </div>
                    ) : openManusProjects && openManusProjects.length > 0 ? (
                      <div className="space-y-4">
                        {openManusProjects.map((project: any) => (
                          <Card 
                            key={project.project_id} 
                            className={`border-l-4 border-l-indigo-500 cursor-pointer hover:shadow-md transition-all`}
                          >
                            <CardContent className="p-4">
                              <div className="flex justify-between items-start">
                                <div>
                                  <h3 className="font-bold">{project.project_name}</h3>
                                  <p className="text-sm text-muted-foreground">{project.requirements}</p>
                                </div>
                                <div className="flex gap-2">
                                  <Badge className={getStatusColor(project.status)}>
                                    {project.status.replace('_', ' ')}
                                  </Badge>
                                </div>
                              </div>
                              <div className="mt-2">
                                <Progress value={project.progress || 0} className="h-2" />
                              </div>
                              <div className="text-xs text-muted-foreground mt-2">
                                Started: {project.started_at ? formatTime(project.started_at) : 'Not started yet'}
                              </div>
                              <div className="mt-2 flex flex-wrap gap-1">
                                <Badge variant="outline">Language: {project.language || 'Not specified'}</Badge>
                                <Badge variant="outline">Framework: {project.framework || 'Not specified'}</Badge>
                                <Badge variant="outline">Tasks: {project.subtask_counts?.total || 0}</Badge>
                                <Badge variant="outline" className="bg-green-100">Completed: {project.subtask_counts?.completed || 0}</Badge>
                                <Badge variant="outline" className="bg-yellow-100">In Progress: {project.subtask_counts?.in_progress || 0}</Badge>
                                <Badge variant="outline" className="bg-red-100">Failed: {project.subtask_counts?.failed || 0}</Badge>
                              </div>
                              <div className="mt-2">
                                <Button 
                                  variant="outline" 
                                  size="sm" 
                                  className="w-full"
                                  onClick={() => setSelectedOpenManusProject(project.project_id)}
                                >
                                  <Layers className="mr-2 h-4 w-4" />
                                  View Project Details
                                </Button>
                              </div>
                            </CardContent>
                          </Card>
                        ))}
                      </div>
                    ) : (
                      <div className="text-center p-8 border rounded-lg border-dashed">
                        <p>No OpenManus projects found. Create your first agentic project by selecting "OpenManus" in the "Primary Model" dropdown!</p>
                      </div>
                    )}
                  </CardContent>
                  <CardFooter>
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={() => refetchOpenManusProjects()}
                      className="w-full"
                    >
                      <RefreshCw className="h-4 w-4 mr-2" />
                      Refresh OpenManus Projects
                    </Button>
                  </CardFooter>
                </Card>
                
                {/* OpenManus Project Details */}
                {selectedOpenManusProject && (
                  <Card className="mt-4">
                    <CardHeader>
                      <CardTitle>OpenManus Project Details</CardTitle>
                      <CardDescription>
                        {openManusProjectDetailsLoading ? 'Loading...' : openManusProjectDetails?.project_name}
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      {openManusProjectDetailsLoading ? (
                        <div className="flex items-center justify-center p-8">
                          <Loader2 className="h-8 w-8 animate-spin text-border" />
                        </div>
                      ) : openManusProjectDetails ? (
                        <div className="space-y-4">
                          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <div>
                              <h3 className="font-semibold">Status</h3>
                              <Badge className={getStatusColor(openManusProjectDetails.status)}>
                                {openManusProjectDetails.status.replace('_', ' ')}
                              </Badge>
                            </div>
                            <div>
                              <h3 className="font-semibold">Progress</h3>
                              <div className="flex items-center gap-2">
                                <Progress value={openManusProjectDetails.progress || 0} className="h-2 flex-1" />
                                <span className="text-sm">{openManusProjectDetails.progress || 0}%</span>
                              </div>
                            </div>
                            <div>
                              <h3 className="font-semibold">Tasks</h3>
                              <div className="flex gap-2">
                                <Badge variant="outline" className="bg-green-100">
                                  {openManusProjectDetails.subtask_counts?.completed || 0} completed
                                </Badge>
                                <Badge variant="outline" className="bg-yellow-100">
                                  {openManusProjectDetails.subtask_counts?.in_progress || 0} in progress
                                </Badge>
                                <Badge variant="outline" className="bg-red-100">
                                  {openManusProjectDetails.subtask_counts?.failed || 0} failed
                                </Badge>
                              </div>
                            </div>
                          </div>
                          
                          <div>
                            <h3 className="font-semibold mb-2">Recent Logs</h3>
                            <div className="bg-muted rounded-md p-4 max-h-60 overflow-y-auto">
                              <pre className="text-sm">
                                {openManusProjectDetails.recent_logs?.map((log: any, index: number) => (
                                  <div key={index} className={`mb-1 ${log.level === 'error' ? 'text-red-500' : ''}`}>
                                    [{new Date(log.timestamp).toLocaleTimeString()}] {log.message}
                                  </div>
                                )).reverse() || 'No logs available'}
                              </pre>
                            </div>
                          </div>
                          
                          <div>
                            <Accordion type="single" collapsible className="w-full">
                              <AccordionItem value="files">
                                <AccordionTrigger>
                                  <div className="flex items-center">
                                    <FileText className="mr-2 h-4 w-4" />
                                    Generated Files
                                  </div>
                                </AccordionTrigger>
                                <AccordionContent>
                                  <div className="p-2 bg-muted rounded-md">
                                    {openManusProjectDetails.files ? 
                                      Object.keys(openManusProjectDetails.files).length > 0 ? (
                                        <div className="space-y-2">
                                          {Object.keys(openManusProjectDetails.files).map((filePath, index) => (
                                            <div key={index} className="flex items-center justify-between">
                                              <code className="text-sm">{filePath}</code>
                                              <Button variant="outline" size="sm">
                                                <Eye className="h-3 w-3 mr-1" />
                                                View
                                              </Button>
                                            </div>
                                          ))}
                                        </div>
                                      ) : (
                                        <p className="text-sm">No files generated yet.</p>
                                      )
                                    : <p className="text-sm">No files available.</p>}
                                  </div>
                                </AccordionContent>
                              </AccordionItem>
                            </Accordion>
                          </div>
                        </div>
                      ) : (
                        <div className="text-center p-4">
                          <p>No project details available</p>
                        </div>
                      )}
                    </CardContent>
                    <CardFooter>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => refetchOpenManusProjectDetails()}
                        className="w-full"
                      >
                        <RefreshCw className="h-4 w-4 mr-2" />
                        Refresh Details
                      </Button>
                    </CardFooter>
                  </Card>
                )}
              </TabsContent>
            </Tabs>
          </div>

          {/* Project Details */}
          {selectedProject && (
            <Card>
              <CardHeader>
                <CardTitle>Project Details</CardTitle>
                <CardDescription>
                  {projectDetailsLoading ? 'Loading...' : projectDetails?.name}
                </CardDescription>
              </CardHeader>
              <CardContent>
                {projectDetailsLoading ? (
                  <div className="flex items-center justify-center p-8">
                    <Loader2 className="h-8 w-8 animate-spin text-border" />
                  </div>
                ) : projectDetails ? (
                  <div className="space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div>
                        <h3 className="font-semibold">Status</h3>
                        <Badge className={getStatusColor(projectDetails.status)}>
                          {projectDetails.status.replace('_', ' ')}
                        </Badge>
                      </div>
                      <div>
                        <h3 className="font-semibold">Current Phase</h3>
                        <Badge className={getPhaseColor(projectDetails.currentPhase)}>
                          {projectDetails.currentPhase.replace('_', ' ')}
                        </Badge>
                      </div>
                      <div>
                        <h3 className="font-semibold">Progress</h3>
                        <div className="flex items-center gap-2">
                          <Progress value={projectDetails.progress || 0} className="h-2 flex-1" />
                          <span className="text-sm">{projectDetails.progress || 0}%</span>
                        </div>
                      </div>
                    </div>

                    <div>
                      <h3 className="font-semibold mb-2">Team Assignment</h3>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                        {Object.entries(projectDetails.teamAssignment || {}).map(([role, model]) => (
                          <Badge key={role} variant="outline" className="justify-between">
                            {role}: <span className="ml-1 font-normal">{model}</span>
                          </Badge>
                        ))}
                      </div>
                    </div>

                    <div>
                      <h3 className="font-semibold mb-2">Recent Logs</h3>
                      <div className="bg-muted rounded-md p-4 max-h-60 overflow-y-auto">
                        <pre className="text-sm">
                          {projectDetails.recentLogs?.map((log: any, index: number) => (
                            <div key={index} className={`mb-1 ${log.level === 'error' ? 'text-red-500' : ''}`}>
                              [{new Date(log.timestamp).toLocaleTimeString()}] [{log.phase.replace('_', ' ')}] {log.message}
                            </div>
                          )).reverse() || 'No logs available'}
                        </pre>
                      </div>
                    </div>

                    <div>
                      <h3 className="font-semibold mb-2">Artifacts</h3>
                      <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
                        <Button 
                          variant={projectDetails.artifactsSummary?.hasRequirements ? "default" : "outline"}
                          className="w-full"
                          disabled={!projectDetails.artifactsSummary?.hasRequirements}
                        >
                          Requirements
                        </Button>
                        <Button 
                          variant={projectDetails.artifactsSummary?.hasArchitecture ? "default" : "outline"}
                          className="w-full"
                          disabled={!projectDetails.artifactsSummary?.hasArchitecture}
                        >
                          Architecture
                        </Button>
                        <Button 
                          variant={projectDetails.artifactsSummary?.codeBaseSize > 0 ? "default" : "outline"}
                          className="w-full"
                          disabled={!projectDetails.artifactsSummary?.codeBaseSize}
                        >
                          Code ({projectDetails.artifactsSummary?.codeBaseSize || 0} files)
                        </Button>
                        <Button 
                          variant={projectDetails.artifactsSummary?.hasTestResults ? "default" : "outline"}
                          className="w-full"
                          disabled={!projectDetails.artifactsSummary?.hasTestResults}
                        >
                          Tests
                        </Button>
                        <Button 
                          variant={projectDetails.artifactsSummary?.hasDocumentation ? "default" : "outline"}
                          className="w-full"
                          disabled={!projectDetails.artifactsSummary?.hasDocumentation}
                        >
                          Documentation
                        </Button>
                      </div>
                    </div>

                    {projectDetails.qualityMetrics && (
                      <div>
                        <h3 className="font-semibold mb-2">Quality Metrics</h3>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                          {Object.entries(projectDetails.qualityMetrics).map(([key, value]) => (
                            <div key={key} className="bg-muted rounded-md p-3">
                              <div className="text-xs text-muted-foreground">{key.replace(/([A-Z])/g, ' $1').replace(/_/g, ' ').trim()}</div>
                              <div className="text-xl font-bold">{value}</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-center p-8">
                    <p>Project details not available</p>
                  </div>
                )}
              </CardContent>
              <CardFooter className="justify-end">
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={() => refetchProjectDetails()}
                >
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Refresh Details
                </Button>
              </CardFooter>
            </Card>
          )}
        </TabsContent>

        {/* Direct AI Coding Tab */}
        <TabsContent value="direct" className="space-y-8">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <Card>
              <CardHeader>
                <CardTitle>Generate Code</CardTitle>
                <CardDescription>
                  Generate code directly from requirements
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Form {...codeGenForm}>
                  <form onSubmit={codeGenForm.handleSubmit(onCodeGenSubmit)} className="space-y-4">
                    <FormField
                      control={codeGenForm.control}
                      name="requirements"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Requirements</FormLabel>
                          <FormControl>
                            <Textarea 
                              placeholder="Describe what code you want to generate" 
                              className="min-h-[150px]" 
                              {...field} 
                            />
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <FormField
                        control={codeGenForm.control}
                        name="language"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Language</FormLabel>
                            <FormControl>
                              <Input placeholder="e.g. JavaScript, Python" {...field} />
                            </FormControl>
                          </FormItem>
                        )}
                      />

                      <FormField
                        control={codeGenForm.control}
                        name="framework"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Framework</FormLabel>
                            <FormControl>
                              <Input placeholder="e.g. React, Django" {...field} />
                            </FormControl>
                          </FormItem>
                        )}
                      />
                    </div>

                    <FormField
                      control={codeGenForm.control}
                      name="primaryModel"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Model</FormLabel>
                          <Select 
                            onValueChange={field.onChange} 
                            defaultValue={field.value}
                          >
                            <FormControl>
                              <SelectTrigger>
                                <SelectValue placeholder="Select a model" />
                              </SelectTrigger>
                            </FormControl>
                            <SelectContent>
                              <SelectItem value="hybrid">Hybrid (Both Models)</SelectItem>
                              <SelectItem value="qwen2.5-7b-omni">Qwen2.5-7b-omni</SelectItem>
                              <SelectItem value="olympiccoder-7b">OlympicCoder-7B</SelectItem>
                            </SelectContent>
                          </Select>
                        </FormItem>
                      )}
                    />

                    <Button 
                      type="submit" 
                      className="w-full"
                      disabled={generateCodeMutation.isPending}
                    >
                      {generateCodeMutation.isPending ? (
                        <>
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          Generating Code...
                        </>
                      ) : (
                        'Generate Code'
                      )}
                    </Button>
                  </form>
                </Form>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Enhance Code</CardTitle>
                <CardDescription>
                  Optimize, refactor, document, test, or fix existing code
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Form {...enhanceForm}>
                  <form onSubmit={enhanceForm.handleSubmit(onEnhanceSubmit)} className="space-y-4">
                    <FormField
                      control={enhanceForm.control}
                      name="code"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Code</FormLabel>
                          <FormControl>
                            <Textarea 
                              placeholder="Paste your code here" 
                              className="min-h-[200px] font-mono" 
                              {...field} 
                            />
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <FormField
                        control={enhanceForm.control}
                        name="enhancement"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Enhancement Type</FormLabel>
                            <Select 
                              onValueChange={field.onChange} 
                              defaultValue={field.value}
                            >
                              <FormControl>
                                <SelectTrigger>
                                  <SelectValue placeholder="Select enhancement" />
                                </SelectTrigger>
                              </FormControl>
                              <SelectContent>
                                <SelectItem value="optimize">Optimize</SelectItem>
                                <SelectItem value="refactor">Refactor</SelectItem>
                                <SelectItem value="document">Document</SelectItem>
                                <SelectItem value="test">Add Tests</SelectItem>
                                <SelectItem value="fix">Fix Bugs</SelectItem>
                              </SelectContent>
                            </Select>
                          </FormItem>
                        )}
                      />

                      <FormField
                        control={enhanceForm.control}
                        name="language"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Language</FormLabel>
                            <FormControl>
                              <Input placeholder="e.g. JavaScript, Python" {...field} />
                            </FormControl>
                          </FormItem>
                        )}
                      />
                    </div>

                    <FormField
                      control={enhanceForm.control}
                      name="requirements"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Requirements (Optional)</FormLabel>
                          <FormControl>
                            <Textarea 
                              placeholder="Additional context or requirements" 
                              {...field} 
                            />
                          </FormControl>
                        </FormItem>
                      )}
                    />

                    <Button 
                      type="submit" 
                      className="w-full"
                      disabled={enhanceCodeMutation.isPending}
                    >
                      {enhanceCodeMutation.isPending ? (
                        <>
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          Enhancing Code...
                        </>
                      ) : (
                        `${enhanceForm.getValues('enhancement')}e Code`
                      )}
                    </Button>
                  </form>
                </Form>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* System Status Tab */}
        <TabsContent value="status" className="space-y-8">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <Card>
              <CardHeader>
                <CardTitle>Model Status</CardTitle>
                <CardDescription>
                  Status of the AI models
                </CardDescription>
              </CardHeader>
              <CardContent>
                {modelsLoading ? (
                  <div className="flex items-center justify-center p-8">
                    <Loader2 className="h-8 w-8 animate-spin text-border" />
                  </div>
                ) : modelStatus ? (
                  <div className="space-y-4">
                    {Object.entries(modelStatus).map(([modelName, modelData]: [string, any]) => (
                      <div key={modelName} className="border rounded-lg p-4">
                        <div className="flex justify-between items-center mb-2">
                          <h3 className="font-bold">{modelName}</h3>
                          <Badge 
                            className={
                              modelData.status === 'ready' 
                                ? 'bg-green-500' 
                                : modelData.status === 'busy' 
                                ? 'bg-yellow-500' 
                                : 'bg-red-500'
                            }
                          >
                            {modelData.status}
                          </Badge>
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-sm">
                          <div>
                            <span className="text-muted-foreground">Requests handled:</span>{' '}
                            {modelData.stats.requestsHandled}
                          </div>
                          <div>
                            <span className="text-muted-foreground">Failed requests:</span>{' '}
                            {modelData.stats.failedRequests}
                          </div>
                          <div>
                            <span className="text-muted-foreground">Avg response time:</span>{' '}
                            {Math.round(modelData.stats.averageResponseTime || 0)}ms
                          </div>
                          <div>
                            <span className="text-muted-foreground">Uptime:</span>{' '}
                            {Math.round(modelData.stats.uptimeSeconds / 60)} min
                          </div>
                        </div>
                        {modelData.stats.lastError && (
                          <div className="mt-2 text-sm text-red-500">
                            Last error: {modelData.stats.lastError}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center p-8 border rounded-lg border-dashed">
                    <p>Model status information not available</p>
                  </div>
                )}
              </CardContent>
              <CardFooter>
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={() => queryClient.invalidateQueries({ queryKey: ['/api/ai/models/status'] })}
                >
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Refresh Status
                </Button>
              </CardFooter>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>System Logs</CardTitle>
                <CardDescription>
                  Recent system activity
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="bg-muted rounded-md p-4 h-96 overflow-y-auto">
                  <div className="font-mono text-sm">
                    <div className="text-green-500">[System] Seren AI System initialized</div>
                    <div>[Qwen2.5-7b-omni] Model loaded successfully</div>
                    <div>[OlympicCoder-7B] Model loaded successfully</div>
                    <div className="text-yellow-500">[Warning] PyTorch acceleration not available, using CPU</div>
                    <div>[System] Continuous execution system ready</div>
                    <div>[System] Neurosymbolic reasoning system initialized</div>
                    <div>[System] Liquid neural network initialized</div>
                    <div>[System] Knowledge library initialized with baseline data</div>
                    <div>[System] Communication system ready for model collaboration</div>
                    <div className="text-blue-500">[Info] Dev team ready for autonomous software generation</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>About Seren AI</CardTitle>
              <CardDescription>
                The autonomous AI development system
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <p>
                Seren is a bleeding-edge autonomous AI development platform that combines 
                the strengths of multiple large language models to create a hyperintelligent 
                dev team capable of generating production-ready software from natural language 
                requirements.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div>
                  <h3 className="text-lg font-semibold mb-2">Core Components</h3>
                  <ul className="list-disc pl-5 space-y-1">
                    <li>Multi-model AI architecture (Qwen2.5-7b-omni + OlympicCoder-7B)</li>
                    <li>Neuro-symbolic reasoning framework</li>
                    <li>Liquid neural networks for adaptive intelligence</li>
                    <li>Metacognitive system for self-improvement</li>
                    <li>Knowledge library system for learning and sharing</li>
                    <li>Continuous execution pipeline</li>
                  </ul>
                </div>

                <div>
                  <h3 className="text-lg font-semibold mb-2">Features</h3>
                  <ul className="list-disc pl-5 space-y-1">
                    <li>Fully autonomous software development</li>
                    <li>Multiple collaboration modes between models</li>
                    <li>Sophisticated reasoning and problem-solving</li>
                    <li>Code generation, enhancement, and review</li>
                    <li>Architecture design and optimization</li>
                    <li>Comprehensive testing and documentation</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}