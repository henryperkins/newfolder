import { api } from './api';
import {
  LoginRequest,
  LoginResponse,
  RegisterRequest,
  ForgotPasswordRequest,
  ResetPasswordRequest,
  ChangePasswordRequest,
  UpdateUserRequest,
  User,
  MessageResponse,
  RegistrationAvailableResponse,
} from '@/types';

export const authApi = {
  checkRegistrationAvailable: (): Promise<RegistrationAvailableResponse> =>
    api.get('/auth/registration-available').then((res) => res.data),

  register: (data: RegisterRequest): Promise<User> =>
    api.post('/auth/register', data).then((res) => res.data),

  login: (data: LoginRequest): Promise<LoginResponse> =>
    api.post('/auth/login', data).then((res) => res.data),

  logout: (): Promise<MessageResponse> =>
    api.post('/auth/logout').then((res) => res.data),

  forgotPassword: (data: ForgotPasswordRequest): Promise<MessageResponse> =>
    api.post('/auth/forgot-password', data).then((res) => res.data),

  resetPassword: (data: ResetPasswordRequest): Promise<MessageResponse> =>
    api.post('/auth/reset-password', data).then((res) => res.data),

  getCurrentUser: (): Promise<User> =>
    api.get('/users/me').then((res) => res.data),

  updateProfile: (data: UpdateUserRequest): Promise<User> =>
    api.patch('/users/me', data).then((res) => res.data),

  changePassword: (data: ChangePasswordRequest): Promise<MessageResponse> =>
    api.post('/users/me/change-password', data).then((res) => res.data),
};