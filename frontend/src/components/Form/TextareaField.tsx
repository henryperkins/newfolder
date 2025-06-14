import React from 'react';
import { useFormContext } from 'react-hook-form';
import { Textarea, TextareaProps } from '@/components/common';

interface FormTextareaProps extends Omit<TextareaProps, 'name' | 'error'> {
  name: string;
}

export const TextareaField: React.FC<FormTextareaProps> = ({
  name,
  ...rest
}) => {
  const {
    register,
    formState: { errors },
  } = useFormContext();

  const error = errors[name]?.message as string | undefined;

  return <Textarea {...register(name)} {...rest} error={error} />;
};
