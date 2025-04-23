import passport from "passport";
import { Strategy as LocalStrategy } from "passport-local";
import { Express, Request, Response, NextFunction } from "express";
import session from "express-session";
import { scrypt, randomBytes, timingSafeEqual } from "crypto";
import { promisify } from "util";
import { storage } from "./storage";
import { User } from "@shared/schema";

// Configure global Express types
declare global {
  namespace Express {
    interface User extends User {}
  }
}

// Create async version of scrypt
const scryptAsync = promisify(scrypt);

/**
 * Hash a password for storage
 * @param password Plain text password to hash
 * @returns Hashed password with salt
 */
async function hashPassword(password: string) {
  try {
    const salt = randomBytes(16).toString("hex");
    const buf = (await scryptAsync(password, salt, 64)) as Buffer;
    return `${buf.toString("hex")}.${salt}`;
  } catch (error) {
    console.error("Password hashing error:", error);
    throw new Error("Failed to securely hash password");
  }
}

/**
 * Compare a supplied password against a stored password
 * @param supplied Plain text password to check
 * @param stored Stored hashed password
 * @returns Boolean indicating if passwords match
 */
async function comparePasswords(supplied: string, stored: string) {
  try {
    const [hashed, salt] = stored.split(".");
    if (!hashed || !salt) {
      console.error("Invalid stored password format");
      return false;
    }
    
    const hashedBuf = Buffer.from(hashed, "hex");
    const suppliedBuf = (await scryptAsync(supplied, salt, 64)) as Buffer;
    return timingSafeEqual(hashedBuf, suppliedBuf);
  } catch (error) {
    console.error("Password comparison error:", error);
    return false;
  }
}

/**
 * Error-handling middleware for authentication errors
 */
function handleAuthError(err: Error, req: Request, res: Response, next: NextFunction) {
  console.error("Authentication error:", err);
  res.status(500).json({ 
    message: "Authentication system error",
    error: process.env.NODE_ENV === "development" ? err.message : "Internal server error"
  });
}

/**
 * Set up authentication for the application
 * @param app Express application
 */
export function setupAuth(app: Express) {
  // Generate a secure session secret or use environment variable
  const sessionSecret = process.env.SESSION_SECRET || randomBytes(32).toString("hex");
  
  // Configure session
  const sessionSettings: session.SessionOptions = {
    secret: sessionSecret,
    resave: false,
    saveUninitialized: false,
    store: storage.sessionStore,
    cookie: {
      maxAge: 24 * 60 * 60 * 1000, // 24 hours
      secure: process.env.NODE_ENV === "production",
      sameSite: "lax",
      httpOnly: true
    }
  };

  // Set up session and passport
  app.set("trust proxy", 1);
  app.use(session(sessionSettings));
  app.use(passport.initialize());
  app.use(passport.session());

  // Configure local strategy for username/password authentication
  passport.use(
    new LocalStrategy(async (username, password, done) => {
      try {
        // Allow both username and email login
        const user = await storage.getUserByUsername(username);
        
        // Check credentials
        if (!user) {
          console.log(`Login failed: User '${username}' not found`);
          return done(null, false, { message: "Incorrect username or password" });
        }
        
        if (!(await comparePasswords(password, user.password))) {
          console.log(`Login failed: Incorrect password for user '${username}'`);
          return done(null, false, { message: "Incorrect username or password" });
        }
        
        // Successful login
        console.log(`User '${username}' authenticated successfully`);
        return done(null, user);
      } catch (error) {
        console.error(`Authentication error for user '${username}':`, error);
        return done(error);
      }
    }),
  );

  // Serialize user to session
  passport.serializeUser((user, done) => {
    done(null, user.id);
  });

  // Deserialize user from session
  passport.deserializeUser(async (id: number, done) => {
    try {
      const user = await storage.getUser(id);
      if (!user) {
        return done(null, false);
      }
      done(null, user);
    } catch (error) {
      console.error(`Error deserializing user ID ${id}:`, error);
      done(error);
    }
  });

  // Registration endpoint
  app.post("/api/register", async (req, res, next) => {
    try {
      // Validate required fields
      if (!req.body.username || !req.body.password) {
        return res.status(400).json({ message: "Username and password are required" });
      }
      
      // Check if username is already taken
      const existingUser = await storage.getUserByUsername(req.body.username);
      if (existingUser) {
        return res.status(400).json({ message: "Username already exists" });
      }

      // Create user with hashed password
      const user = await storage.createUser({
        ...req.body,
        password: await hashPassword(req.body.password),
      });

      // Log user in automatically
      req.login(user, (err) => {
        if (err) return next(err);
        // Return user without password
        const { password, ...safeUser } = user;
        res.status(201).json(safeUser);
      });
    } catch (error) {
      console.error("Registration error:", error);
      // Pass to error handler
      next(error);
    }
  });

  // Login endpoint
  app.post("/api/login", 
    (req, res, next) => {
      if (!req.body.username || !req.body.password) {
        return res.status(400).json({ message: "Username and password are required" });
      }
      next();
    },
    passport.authenticate("local", { failWithError: true }),
    (req, res) => {
      // Return user without password
      const { password, ...safeUser } = req.user as User;
      res.status(200).json(safeUser);
    },
    (err: Error, req: Request, res: Response, next: NextFunction) => {
      return res.status(401).json({ message: "Authentication failed", error: err.message });
    }
  );

  // Logout endpoint
  app.post("/api/logout", (req, res, next) => {
    // Check if user is authenticated
    if (req.isAuthenticated()) {
      req.logout((err) => {
        if (err) return next(err);
        res.status(200).json({ message: "Logged out successfully" });
      });
    } else {
      res.status(200).json({ message: "Not logged in" });
    }
  });

  // Get current user endpoint
  app.get("/api/user", (req, res) => {
    if (!req.isAuthenticated()) {
      return res.status(401).json({ message: "Not authenticated" });
    }
    // Return user without password
    const { password, ...safeUser } = req.user as User;
    res.json(safeUser);
  });
  
  // Create admin user if it doesn't exist and initialize settings
  (async () => {
    try {
      console.log("Checking for admin user...");
      const adminExists = await storage.getUserByUsername("admin");
      let adminUser: User;
      
      if (!adminExists) {
        console.log("Creating admin user...");
        adminUser = await storage.createUser({
          username: "admin",
          password: await hashPassword("admin123"),
          isAdmin: true
        });
        console.log("✅ Created admin user (username: admin, password: admin123)");
      } else {
        console.log("Admin user already exists");
        adminUser = adminExists;
      }
      
      // Initialize settings with admin user ID
      console.log("Initializing application settings...");
      await storage.initializeSettings(adminUser.id);
    } catch (error) {
      console.error("Failed to set up admin account:", error);
    }
  })();
  
  // Register error handler
  app.use("/api/register", handleAuthError);
  app.use("/api/login", handleAuthError);
  app.use("/api/logout", handleAuthError);
}
