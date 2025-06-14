import React from 'react';
import { useFormContext } from 'react-hook-form';
import { Input } from '@/components/common';
import { InputProps } from '@/components/common/Input';

interface FormInputProps extends Omit<InputProps, 'name' | 'error'> {
  name: string;
}

export const InputField: React.FC<FormInputProps> = ({ name, ...rest }) => {
  const {
    register,
    formState: { errors },
  } = useFormContext();

  const error = errors[name]?.message as string | undefined;

  return <Input {...register(name)} {...rest} error={error} />;
};
