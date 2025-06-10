import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { User, Lock } from 'lucide-react';
import { Button, Input, Card } from '@/components/common';
import { useAuthStore } from '@/stores';
import { authApi } from '@/utils';
import { UpdateUserRequest, ChangePasswordRequest } from '@/types';

const updateProfileSchema = z.object({
  username: z
    .string()
    .min(3, 'Username must be at least 3 characters')
    .max(30, 'Username must be less than 30 characters')
    .regex(/^[a-zA-Z0-9_]+$/, 'Username can only contain letters, numbers, and underscores'),
  email: z.string().email('Invalid email format'),
});

const changePasswordSchema = z.object({
  currentPassword: z.string().min(1, 'Current password is required'),
  newPassword: z
    .string()
    .min(8, 'Password must be at least 8 characters')
    .regex(/[A-Z]/, 'Password must contain an uppercase letter')
    .regex(/[a-z]/, 'Password must contain a lowercase letter')
    .regex(/[0-9]/, 'Password must contain a number'),
  confirmPassword: z.string(),
}).refine((data) => data.newPassword === data.confirmPassword, {
  message: "Passwords don't match",
  path: ['confirmPassword'],
});

type UpdateProfileFormData = z.infer<typeof updateProfileSchema>;
type ChangePasswordFormData = z.infer<typeof changePasswordSchema>;

export const SettingsPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState('profile');
  const [isUpdatingProfile, setIsUpdatingProfile] = useState(false);
  const [isChangingPassword, setIsChangingPassword] = useState(false);
  const [profileSuccess, setProfileSuccess] = useState(false);
  const [passwordSuccess, setPasswordSuccess] = useState(false);
  const [profileError, setProfileError] = useState<string | null>(null);
  const [passwordError, setPasswordError] = useState<string | null>(null);
  
  const { user, setUser } = useAuthStore();

  const profileForm = useForm<UpdateProfileFormData>({
    resolver: zodResolver(updateProfileSchema),
    defaultValues: {
      username: user?.username || '',
      email: user?.email || '',
    },
  });

  const passwordForm = useForm<ChangePasswordFormData>({
    resolver: zodResolver(changePasswordSchema),
  });

  const onUpdateProfile = async (data: UpdateProfileFormData) => {
    setIsUpdatingProfile(true);
    setProfileError(null);
    setProfileSuccess(false);

    try {
      const updatedUser = await authApi.updateProfile(data);
      setUser(updatedUser);
      setProfileSuccess(true);
      setTimeout(() => setProfileSuccess(false), 3000);
    } catch (err: any) {
      const message = err.response?.data?.detail || 'Failed to update profile. Please try again.';
      setProfileError(message);
    } finally {
      setIsUpdatingProfile(false);
    }
  };

  const onChangePassword = async (data: ChangePasswordFormData) => {
    setIsChangingPassword(true);
    setPasswordError(null);
    setPasswordSuccess(false);

    try {
      await authApi.changePassword({
        current_password: data.currentPassword,
        new_password: data.newPassword,
      });
      setPasswordSuccess(true);
      passwordForm.reset();
      setTimeout(() => setPasswordSuccess(false), 3000);
    } catch (err: any) {
      const message = err.response?.data?.detail || 'Failed to change password. Please try again.';
      setPasswordError(message);
    } finally {
      setIsChangingPassword(false);
    }
  };

  const tabs = [
    { id: 'profile', label: 'Profile Settings', icon: User },
    { id: 'security', label: 'Security', icon: Lock },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Settings</h1>
        <p className="text-gray-600">Manage your account and preferences</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Sidebar Navigation */}
        <div className="lg:col-span-1">
          <Card padding={false}>
            <nav className="space-y-1 p-2">
              {tabs.map((tab) => {
                const Icon = tab.icon;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`w-full flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors ${
                      activeTab === tab.id
                        ? 'bg-primary-50 text-primary-700'
                        : 'text-gray-700 hover:bg-gray-100'
                    }`}
                  >
                    <Icon className="w-5 h-5 mr-3" />
                    {tab.label}
                  </button>
                );
              })}
            </nav>
          </Card>
        </div>

        {/* Content */}
        <div className="lg:col-span-3">
          {activeTab === 'profile' && (
            <Card>
              <div className="space-y-6">
                <div>
                  <h2 className="text-lg font-semibold text-gray-900">Profile Information</h2>
                  <p className="text-gray-600">Update your account's profile information.</p>
                </div>

                <form onSubmit={profileForm.handleSubmit(onUpdateProfile)} className="space-y-4">
                  {profileError && (
                    <div className="p-3 text-sm text-red-600 bg-red-50 border border-red-200 rounded-md">
                      {profileError}
                    </div>
                  )}
                  
                  {profileSuccess && (
                    <div className="p-3 text-sm text-green-600 bg-green-50 border border-green-200 rounded-md">
                      Profile updated successfully!
                    </div>
                  )}

                  <Input
                    {...profileForm.register('username')}
                    label="Username"
                    error={profileForm.formState.errors.username?.message}
                  />

                  <Input
                    {...profileForm.register('email')}
                    type="email"
                    label="Email Address"
                    error={profileForm.formState.errors.email?.message}
                  />

                  <div className="flex justify-end">
                    <Button
                      type="submit"
                      isLoading={isUpdatingProfile}
                      disabled={isUpdatingProfile}
                    >
                      {isUpdatingProfile ? 'Saving...' : 'Save Changes'}
                    </Button>
                  </div>
                </form>
              </div>
            </Card>
          )}

          {activeTab === 'security' && (
            <Card>
              <div className="space-y-6">
                <div>
                  <h2 className="text-lg font-semibold text-gray-900">Change Password</h2>
                  <p className="text-gray-600">Update your password to keep your account secure.</p>
                </div>

                <form onSubmit={passwordForm.handleSubmit(onChangePassword)} className="space-y-4">
                  {passwordError && (
                    <div className="p-3 text-sm text-red-600 bg-red-50 border border-red-200 rounded-md">
                      {passwordError}
                    </div>
                  )}
                  
                  {passwordSuccess && (
                    <div className="p-3 text-sm text-green-600 bg-green-50 border border-green-200 rounded-md">
                      Password changed successfully!
                    </div>
                  )}

                  <Input
                    {...passwordForm.register('currentPassword')}
                    type="password"
                    label="Current Password"
                    error={passwordForm.formState.errors.currentPassword?.message}
                    autoComplete="current-password"
                  />

                  <Input
                    {...passwordForm.register('newPassword')}
                    type="password"
                    label="New Password"
                    error={passwordForm.formState.errors.newPassword?.message}
                    autoComplete="new-password"
                  />

                  <Input
                    {...passwordForm.register('confirmPassword')}
                    type="password"
                    label="Confirm New Password"
                    error={passwordForm.formState.errors.confirmPassword?.message}
                    autoComplete="new-password"
                  />

                  <div className="flex justify-end">
                    <Button
                      type="submit"
                      isLoading={isChangingPassword}
                      disabled={isChangingPassword}
                    >
                      {isChangingPassword ? 'Changing Password...' : 'Change Password'}
                    </Button>
                  </div>
                </form>
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};