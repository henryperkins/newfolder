import React, { useEffect, useState, useCallback } from 'react';
import { 
  FolderPlus, 
  MessageSquare, 
  FileUp, 
  Edit3, 
  Archive, 
  Trash2,
  Clock,
  ChevronDown
} from 'lucide-react';
import { useProjectStore } from '@/stores';
import { ActivityItem, ActivitiesQueryParams } from '@/types';
import { cn } from '@/utils';

interface ActivityTimelineProps {
  projectId?: string;
  limit?: number;
  onLoadMore?: () => void;
}

const activityIcons = {
  project_created: FolderPlus,
  project_updated: Edit3,
  project_archived: Archive,
  project_deleted: Trash2,
  chat_started: MessageSquare,
  document_uploaded: FileUp,
  document_deleted: Trash2,
};

const activityColors = {
  project_created: 'text-green-600 bg-green-100',
  project_updated: 'text-blue-600 bg-blue-100',
  project_archived: 'text-yellow-600 bg-yellow-100',
  project_deleted: 'text-red-600 bg-red-100',
  chat_started: 'text-purple-600 bg-purple-100',
  document_uploaded: 'text-indigo-600 bg-indigo-100',
  document_deleted: 'text-red-600 bg-red-100',
};

export const ActivityTimeline: React.FC<ActivityTimelineProps> = ({
  projectId,
  limit = 50,
  onLoadMore
}) => {
  const [expandedItems, setExpandedItems] = useState<Set<string>>(new Set());

  const {
    activities,
    isLoadingActivities,
    activitiesError,
    hasMoreActivities,
    fetchActivities,
    loadMoreActivities
  } = useProjectStore();

  useEffect(() => {
    const params: ActivitiesQueryParams = {
      project_id: projectId,
      limit
    };
    fetchActivities(params);
  }, [fetchActivities, projectId, limit]);

  const handleLoadMore = useCallback(() => {
    if (onLoadMore) {
      onLoadMore();
    } else {
      loadMoreActivities();
    }
  }, [onLoadMore, loadMoreActivities]);

  const toggleExpanded = (itemId: string) => {
    setExpandedItems(prev => {
      const newSet = new Set(prev);
      if (newSet.has(itemId)) {
        newSet.delete(itemId);
      } else {
        newSet.add(itemId);
      }
      return newSet;
    });
  };

  const formatTimeAgo = (dateString: string) => {
    const now = new Date();
    const date = new Date(dateString);
    const diff = now.getTime() - date.getTime();
    
    const minutes = Math.floor(diff / (1000 * 60));
    const hours = Math.floor(diff / (1000 * 60 * 60));
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));

    if (minutes < 1) {
      return 'Just now';
    } else if (minutes < 60) {
      return `${minutes}m ago`;
    } else if (hours < 24) {
      return `${hours}h ago`;
    } else if (days < 7) {
      return `${days}d ago`;
    } else {
      return date.toLocaleDateString();
    }
  };

  const groupActivitiesByDay = (activities: ActivityItem[]) => {
    const groups: Record<string, ActivityItem[]> = {};
    
    activities.forEach(activity => {
      const date = new Date(activity.created_at);
      const dayKey = date.toDateString();
      
      if (!groups[dayKey]) {
        groups[dayKey] = [];
      }
      groups[dayKey].push(activity);
    });

    return Object.entries(groups).map(([day, items]) => ({
      day,
      items
    }));
  };

  const getActivityDescription = (activity: ActivityItem): string => {
    const baseDescriptions = {
      project_created: 'Created project',
      project_updated: 'Updated project',
      project_archived: 'Archived project',
      project_deleted: 'Deleted project',
      chat_started: 'Started a chat',
      document_uploaded: 'Uploaded document',
      document_deleted: 'Deleted document',
    };

    let description = baseDescriptions[activity.activity_type as keyof typeof baseDescriptions] || 'Unknown activity';
    
    if (activity.project_name) {
      description += ` in ${activity.project_name}`;
    }

    return description;
  };

  const hasMetadata = (activity: ActivityItem): boolean => {
    return Object.keys(activity.metadata).length > 0;
  };

  if (isLoadingActivities && activities.length === 0) {
    return (
      <div className="space-y-6">
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="flex gap-4 animate-pulse">
            <div className="w-8 h-8 bg-gray-200 rounded-full flex-shrink-0 mt-1" />
            <div className="flex-1">
              <div className="h-4 bg-gray-200 rounded w-3/4 mb-2" />
              <div className="h-3 bg-gray-200 rounded w-1/2" />
            </div>
          </div>
        ))}
      </div>
    );
  }

  if (activitiesError) {
    return (
      <div className="text-center py-8">
        <div className="text-red-600 mb-2">Failed to load activities</div>
        <button
          onClick={() => fetchActivities()}
          className="text-blue-600 hover:text-blue-700 text-sm"
        >
          Try again
        </button>
      </div>
    );
  }

  if (activities.length === 0) {
    return (
      <div className="text-center py-8">
        <Clock className="w-12 h-12 text-gray-300 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-gray-900 mb-2">
          No recent activity
        </h3>
        <p className="text-gray-600">
          {projectId 
            ? 'Activity in this project will appear here'
            : 'Your recent actions will appear here'
          }
        </p>
      </div>
    );
  }

  const groupedActivities = groupActivitiesByDay(activities);

  return (
    <div className="space-y-6">
      {groupedActivities.map(({ day, items }, groupIndex) => (
        <div key={day}>
          {/* Day Header */}
          <div className="relative mb-4">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-gray-200" />
            </div>
            <div className="relative flex justify-center">
              <span className="bg-white px-3 text-sm font-medium text-gray-500">
                {new Date(day).toLocaleDateString(undefined, { 
                  weekday: 'long', 
                  month: 'long', 
                  day: 'numeric',
                  year: new Date(day).getFullYear() !== new Date().getFullYear() ? 'numeric' : undefined
                })}
              </span>
            </div>
          </div>

          {/* Activities */}
          <div className="space-y-4">
            {items.map((activity, index) => {
              const IconComponent = activityIcons[activity.activity_type as keyof typeof activityIcons] || Clock;
              const colorClasses = activityColors[activity.activity_type as keyof typeof activityColors] || 'text-gray-600 bg-gray-100';
              const isExpanded = expandedItems.has(activity.id);
              const hasDetails = hasMetadata(activity);

              return (
                <div key={activity.id} className="relative flex gap-4">
                  {/* Timeline Line */}
                  {!(groupIndex === groupedActivities.length - 1 && index === items.length - 1) && (
                    <div className="absolute left-4 top-8 bottom-0 w-px bg-gray-200" />
                  )}

                  {/* Icon */}
                  <div className={cn(
                    'w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 mt-1',
                    colorClasses
                  )}>
                    <IconComponent className="w-4 h-4" />
                  </div>

                  {/* Content */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-start justify-between">
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-gray-900">
                          {getActivityDescription(activity)}
                        </p>
                        <p className="text-xs text-gray-500 mt-1">
                          {formatTimeAgo(activity.created_at)}
                        </p>
                      </div>

                      {/* Expand Button */}
                      {hasDetails && (
                        <button
                          onClick={() => toggleExpanded(activity.id)}
                          className="ml-2 p-1 text-gray-400 hover:text-gray-600 transition-colors"
                        >
                          <ChevronDown className={cn(
                            'w-4 h-4 transition-transform',
                            isExpanded && 'rotate-180'
                          )} />
                        </button>
                      )}
                    </div>

                    {/* Expanded Details */}
                    {isExpanded && hasDetails && (
                      <div className="mt-3 p-3 bg-gray-50 rounded-md">
                        <h4 className="text-xs font-medium text-gray-700 mb-2 uppercase tracking-wide">
                          Details
                        </h4>
                        <div className="space-y-1">
                          {Object.entries(activity.metadata).map(([key, value]) => (
                            <div key={key} className="flex justify-between text-xs">
                              <span className="text-gray-600 capitalize">
                                {key.replace(/_/g, ' ')}:
                              </span>
                              <span className="text-gray-900 font-mono">
                                {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      ))}

      {/* Load More */}
      {hasMoreActivities && (
        <div className="text-center pt-4">
          <button
            onClick={handleLoadMore}
            disabled={isLoadingActivities}
            className="px-4 py-2 text-sm font-medium text-blue-600 bg-blue-50 border border-blue-200 rounded-md hover:bg-blue-100 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isLoadingActivities ? 'Loading...' : 'Load More'}
          </button>
        </div>
      )}
    </div>
  );
};