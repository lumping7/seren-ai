import * as React from "react";
import { cn } from "@/lib/utils";

interface CustomCheckboxProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
}

export const CustomCheckbox = React.forwardRef<HTMLInputElement, CustomCheckboxProps>(
  ({ className, label, ...props }, ref) => {
    return (
      <div className="flex items-center">
        <label className="switch">
          <input
            type="checkbox"
            ref={ref}
            {...props}
          />
          <span className="slider"></span>
        </label>
        {label && <span className="ml-2 text-sm text-neutral-700">{label}</span>}
      </div>
    );
  }
);

CustomCheckbox.displayName = "CustomCheckbox";
