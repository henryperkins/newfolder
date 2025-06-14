import React from 'react';
import {
  useForm,
  FormProvider,
  SubmitHandler,
  FieldValues,
  UseFormProps,
} from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';

interface FormProps<T extends FieldValues>
  extends Omit<React.FormHTMLAttributes<HTMLFormElement>, 'onSubmit'> {
  validationSchema: z.Schema<T>;
  onSubmit: SubmitHandler<T>;
  formOptions?: UseFormProps<T>;
  children: React.ReactNode;
}

export const Form = <T extends FieldValues>({
  validationSchema,
  onSubmit,
  formOptions,
  children,
  ...rest
}: FormProps<T>) => {
  const methods = useForm<T>({
    ...formOptions,
    resolver: zodResolver(validationSchema),
  });

  return (
    <FormProvider {...methods}>
      <form onSubmit={methods.handleSubmit(onSubmit)} {...rest}>
        {children}
      </form>
    </FormProvider>
  );
};
