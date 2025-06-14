import React, { useState, useRef, useEffect } from 'react';
import { Bell, FileText, CheckCircle, AlertCircle } from 'lucide-react';
import { useDocumentStore } from '@/stores/documentStore';
import { cn, formatRelativeTime } from '@/utils';

export const NotificationBell: React.FC = () => {
    const [showDropdown, setShowDropdown] = useState(false);
    const dropdownRef = useRef < HTMLDivElement > (null);

    const {
        notifications,
        unreadNotificationCount,
        markNotificationsRead,
        clearNotifications
    } = useDocumentStore();

    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
                setShowDropdown(false);
            }
        };

        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    const handleBellClick = () => {
        setShowDropdown(!showDropdown);
        if (!showDropdown && unreadNotificationCount > 0) {
            markNotificationsRead();
        }
    };

    const getIcon = (type: string) => {
        switch (type) {
            case 'success':
                return <CheckCircle className="w-5 h-5 text-green-600" />;
            case 'error':
                return <AlertCircle className="w-5 h-5 text-red-600" />;
            default:
                return <FileText className="w-5 h-5 text-blue-600" />;
        }
    };

    return (
        <div className="relative" ref={dropdownRef}>
            <button
                onClick={handleBellClick}
                className="relative p-2 hover:bg-gray-100 rounded-lg transition-colors"
            >
                <Bell className="w-5 h-5 text-gray-600" />
                {unreadNotificationCount > 0 && (
                    <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full"></span>
                )}
            </button>

            {showDropdown && (
                <div className="absolute right-0 top-full mt-2 w-96 bg-white rounded-lg shadow-lg border max-h-96 overflow-hidden z-50">
                    {/* Header */}
                    <div className="p-4 border-b flex items-center justify-between">
                        <h3 className="font-semibold text-gray-900">Document Notifications</h3>
                        {notifications.length > 0 && (
                            <button
                                onClick={clearNotifications}
                                className="text-sm text-gray-500 hover:text-gray-700"
                            >
                                Clear all
                            </button>
                        )}
                    </div>

                    {/* Notification List */}
                    <div className="overflow-y-auto max-h-80">
                        {notifications.length === 0 ? (
                            <div className="p-8 text-center text-gray-500">
                                <Bell className="w-8 h-8 mx-auto mb-3 text-gray-300" />
                                <p>No notifications</p>
                            </div>
                        ) : (
                            <div className="divide-y">
                                {notifications.map((notification) => (
                                    <div
                                        key={notification.id}
                                        className={cn(
                                            'p-4 hover:bg-gray-50 transition-colors',
                                            notification.type === 'error' && 'bg-red-50'
                                        )}
                                    >
                                        <div className="flex gap-3">
                                            <div className="flex-shrink-0">
                                                {getIcon(notification.type)}
                                            </div>
                                            <div className="flex-1 min-w-0">
                                                <p className="text-sm font-medium text-gray-900 truncate">
                                                    {notification.documentName}
                                                </p>
                                                <p className="text-sm text-gray-600 mt-1">
                                                    {notification.message}
                                                </p>
                                                <p className="text-xs text-gray-500 mt-1">
                                                    {formatRelativeTime(notification.timestamp)}
                                                </p>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};
