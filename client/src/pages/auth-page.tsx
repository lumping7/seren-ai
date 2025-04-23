import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { useAuth, loginSchema, registerSchema } from "@/hooks/use-auth";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useState } from "react";
import { z } from "zod";
import { Redirect } from "wouter";
import { Loader2 } from "lucide-react";

type AuthFormType = "login" | "register";

export default function AuthPage() {
  const [formType, setFormType] = useState<AuthFormType>("login");
  const { user, loginMutation, registerMutation } = useAuth();

  // Redirect if already logged in
  if (user) {
    return <Redirect to="/" />;
  }

  return (
    <div className="w-full min-h-screen flex flex-col lg:flex-row">
      {/* Auth form */}
      <div className="flex flex-col justify-center p-8 lg:w-1/2">
        <div className="mx-auto w-full max-w-md space-y-6">
          <div className="space-y-2 text-center">
            <h1 className="text-3xl font-bold">
              {formType === "login" ? "Welcome back" : "Create an account"}
            </h1>
            <p className="text-gray-500 dark:text-gray-400">
              {formType === "login"
                ? "Enter your credentials to access your account"
                : "Enter your information to create an account"}
            </p>
          </div>

          {formType === "login" ? (
            <LoginForm onSuccess={() => {}} onToggle={() => setFormType("register")} />
          ) : (
            <RegisterForm onSuccess={() => {}} onToggle={() => setFormType("login")} />
          )}
        </div>
      </div>

      {/* Hero section */}
      <div className="hidden lg:flex lg:w-1/2 bg-gradient-to-br from-indigo-950 via-purple-900 to-black">
        <div className="flex flex-col justify-center items-center h-full p-12 text-white">
          <div className="max-w-md text-center space-y-6">
            <h2 className="text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-indigo-300">
              Seren AI
            </h2>
            <p className="text-lg text-gray-300">
              A next-generation autonomous AI development platform that enables intelligent system creation through advanced technological integrations and adaptive AI workflows.
            </p>
            <div className="pt-4">
              <h3 className="text-xl font-semibold mb-3 text-indigo-300">Core Features</h3>
              <ul className="space-y-2 text-left">
                <li className="flex items-start">
                  <span className="mr-2">✦</span>
                  <span>Multi-model AI architecture (Qwen2.5, OlympicCoder-7B)</span>
                </li>
                <li className="flex items-start">
                  <span className="mr-2">✦</span>
                  <span>Advanced neuro-symbolic reasoning framework</span>
                </li>
                <li className="flex items-start">
                  <span className="mr-2">✦</span>
                  <span>Military-grade security and encryption</span>
                </li>
                <li className="flex items-start">
                  <span className="mr-2">✦</span>
                  <span>Fully autonomous software development</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function LoginForm({ onSuccess, onToggle }: { onSuccess: () => void; onToggle: () => void }) {
  const { loginMutation } = useAuth();

  const form = useForm<z.infer<typeof loginSchema>>({
    resolver: zodResolver(loginSchema),
    defaultValues: {
      username: "",
      password: "",
    },
  });

  function onSubmit(values: z.infer<typeof loginSchema>) {
    loginMutation.mutate(values, {
      onSuccess,
    });
  }

  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
        <FormField
          control={form.control}
          name="username"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Username</FormLabel>
              <FormControl>
                <Input placeholder="yourusername" {...field} />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        <FormField
          control={form.control}
          name="password"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Password</FormLabel>
              <FormControl>
                <Input type="password" placeholder="••••••••" {...field} />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        <Button 
          className="w-full" 
          type="submit" 
          variant="gradient"
          disabled={loginMutation.isPending}
        >
          {loginMutation.isPending ? (
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          ) : null}
          Sign In
        </Button>
        <div className="text-center text-sm">
          Don't have an account?{" "}
          <Button variant="link" onClick={onToggle} className="p-0">
            Sign up
          </Button>
        </div>
      </form>
    </Form>
  );
}

function RegisterForm({ onSuccess, onToggle }: { onSuccess: () => void; onToggle: () => void }) {
  const { registerMutation } = useAuth();

  const form = useForm<z.infer<typeof registerSchema>>({
    resolver: zodResolver(registerSchema),
    defaultValues: {
      username: "",
      password: "",
      email: "",
      displayName: "",
    },
  });

  function onSubmit(values: z.infer<typeof registerSchema>) {
    registerMutation.mutate(values, {
      onSuccess,
    });
  }

  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
        <FormField
          control={form.control}
          name="username"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Username</FormLabel>
              <FormControl>
                <Input placeholder="yourusername" {...field} />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        <FormField
          control={form.control}
          name="displayName"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Display Name</FormLabel>
              <FormControl>
                <Input placeholder="Your Name" {...field} />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        <FormField
          control={form.control}
          name="email"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Email</FormLabel>
              <FormControl>
                <Input type="email" placeholder="your@email.com" {...field} />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        <FormField
          control={form.control}
          name="password"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Password</FormLabel>
              <FormControl>
                <Input type="password" placeholder="••••••••" {...field} />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        <Button 
          className="w-full" 
          type="submit" 
          variant="gradient"
          disabled={registerMutation.isPending}
        >
          {registerMutation.isPending ? (
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          ) : null}
          Create Account
        </Button>
        <div className="text-center text-sm">
          Already have an account?{" "}
          <Button variant="link" onClick={onToggle} className="p-0">
            Sign in
          </Button>
        </div>
      </form>
    </Form>
  );
}