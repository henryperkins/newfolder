import React, { useState, useEffect } from 'react';
import { X } from 'lucide-react';
import { cn } from '@/utils';
import { Toast } from './toastTypes';
import { icons, styles, iconStyles } from './toastConfig';
import { ToastContext } from './ToastContext';



interface ToastItemProps {
  toast: Toast;
  onRemove: (id: string) =>void;
}

const ToastItem: React.FC<ToastItemProps> = ({ toast, onRemove }) => {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    setIsVisible(true);

    if (toast.duration !== 0) {
      const timer = setTimeout(() => {
        setIsVisible(false);
        setTimeout(() => onRemove(toast.id), 300); // Wait for animation
      }, toast.duration || 5000);

      return () => clearTimeout(timer);
    }
  }, [toast.duration, toast.id, onRemove]);

  const Icon = icons[toast.type];

  return (
    <div
      className={cn(
        'transform transition-all duration-300 ease-out',
        isVisible
          ? 'translate-x-0 opacity-100 scale-100'
          : 'translate-x-full opacity-0 scale-95'
      )}
    >
      <div
        className={cn(
          'p-4 rounded-lg border shadow-md max-w-sm w-full',
          styles[toast.type]
        )}
      >
        <div className="flex items-start gap-3">
          <Icon className={cn('w-5 h-5 mt-0.5 flex-shrink-0', iconStyles[toast.type])} />

          <div className="flex-1 min-w-0">
            <h4 className="text-sm font-medium">{toast.title}</h4>
            {toast.message && (
              <p className="text-sm opacity-90 mt-1">{toast.message}</p>
            )}
            {toast.action && (
              <button
                onClick={toast.action.onClick}
                className="text-sm font-medium underline mt-2 hover:no-underline"
              >
                {toast.action.label}
              </button>
            )}
          </div>

          <button
            onClick={() => onRemove(toast.id)}
            className="p-1 hover:bg-black/10 rounded transition-colors flex-shrink-0"
            aria-label="Close toast"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
};

interface ToastProviderProps {
  children: React.ReactNode;
}

export const ToastProvider: React.FC<ToastProviderProps> = ({ children }) => {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const showToast = (toast: Omit<Toast, 'id'>) => {
    const id = Date.now().toString();
    setToasts(prev => [...prev, { ...toast, id }]);
  };

  const removeToast = (id: string) => {
    setToasts(prev => prev.filter(toast => toast.id !== id));
  };

  return (
    <ToastContext.Provider value={{ showToast, removeToast }}>
      {children}

      {/* Toast Container */}
      <div className="fixed top-4 right-4 z-50 space-y-2">
        {toasts.map(toast => (
          <ToastItem
            key={toast.id}
            toast={toast}
            onRemove={removeToast}
          />
        ))}
      </div>
    </ToastContext.Provider>
  );
};
