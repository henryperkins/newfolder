import React, { useState } from 'react';
import { Plus, Search } from 'lucide-react';
import { Card, Button, Input, Textarea } from '@/components/common';
import {
  Modal,
  Badge,
  Tooltip,
  LoadingSpinner,
  LoadingOverlay,
  Select,
  EmptyState,
  useToast,
} from '@/components/ui';
import { Form, InputField, TextareaField } from '@/components/Form';
import { z } from 'zod';

export const ComponentsShowcasePage: React.FC = () => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [selectValue, setSelectValue] = useState('');
  const { showToast } = useToast();

  const sampleFormSchema = z.object({
    name: z.string().min(1, 'Name is required'),
    email: z.string().email('Invalid email address'),
    message: z.string().min(10, 'Message must be at least 10 characters'),
  });

  type SampleFormType = z.infer<typeof sampleFormSchema>;

  const handleFormSubmit = (data: SampleFormType) => {
    console.log('Form submitted:', data);
    showToast({
      type: 'success',
      title: 'Form Submitted',
      message: JSON.stringify(data, null, 2),
    });
  };

  const selectOptions = [
    { value: 'option1', label: 'Option 1' },
    { value: 'option2', label: 'Option 2' },
    { value: 'option3', label: 'Option 3 (Disabled)', disabled: true },
  ];

  const simulateLoading = () => {
    setIsLoading(true);
    setTimeout(() => setIsLoading(false), 2000);
  };

  return (
    <div className="space-y-8 p-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Components Showcase
        </h1>
        <p className="text-gray-600">
          Preview of all available UI components and their variants
        </p>
      </div>

      {/* Buttons */}
      <Card>
        <h2 className="text-xl font-semibold mb-4">Buttons</h2>
        <div className="space-y-4">
          <div className="flex gap-2 flex-wrap">
            <Button variant="primary">Primary</Button>
            <Button variant="secondary">Secondary</Button>
            <Button variant="ghost">Ghost</Button>
            <Button variant="destructive">Destructive</Button>
            <Button variant="outline">Outline</Button>
          </div>
          <div className="flex gap-2 flex-wrap">
            <Button size="sm">Small</Button>
            <Button size="md">Medium</Button>
            <Button size="lg">Large</Button>
          </div>
          <div className="flex gap-2 flex-wrap">
            <Button isLoading>Loading</Button>
            <Button disabled>Disabled</Button>
          </div>
        </div>
      </Card>

      {/* Badges */}
      <Card>
        <h2 className="text-xl font-semibold mb-4">Badges</h2>
        <div className="flex gap-2 flex-wrap">
          <Badge variant="default">Default</Badge>
          <Badge variant="success">Success</Badge>
          <Badge variant="warning">Warning</Badge>
          <Badge variant="error">Error</Badge>
          <Badge variant="info">Info</Badge>
          <Badge variant="outline">Outline</Badge>
        </div>
      </Card>

      {/* Loading States */}
      <Card>
        <h2 className="text-xl font-semibold mb-4">Loading States</h2>
        <div className="space-y-4">
          <div className="flex gap-4 items-center">
            <LoadingSpinner size="sm" />
            <LoadingSpinner size="md" />
            <LoadingSpinner size="lg" />
            <LoadingSpinner size="xl" />
          </div>
          <LoadingOverlay isLoading={isLoading}>
            <div className="p-8 bg-gray-100 rounded">
              <p>This content can be overlaid with loading state</p>
              <Button onClick={simulateLoading} className="mt-2">
                Simulate Loading
              </Button>
            </div>
          </LoadingOverlay>
        </div>
      </Card>

      {/* Form Elements */}
      <Card>
        <h2 className="text-xl font-semibold mb-4">Form Elements</h2>
        <div className="space-y-4 max-w-md">
          <Input
            label="Text Input"
            placeholder="Enter some text..."
            helperText="This is helper text"
          />
          <Input
            label="Input with Error"
            error="This field is required"
            placeholder="Error state"
          />
          <Select
            label="Select Dropdown"
            options={selectOptions}
            value={selectValue}
            onChange={setSelectValue}
            placeholder="Choose an option..."
          />
          <Textarea
            label="Textarea"
            placeholder="Enter a long text..."
            helperText="This is a textarea component"
          />
          <Textarea
            label="Textarea with Error"
            error="This field is required"
            placeholder="Error state"
          />
        </div>
      </Card>

      {/* Tooltips */}
      <Card>
        <h2 className="text-xl font-semibold mb-4">Tooltips</h2>
        <div className="flex gap-4">
          <Tooltip content="This is a tooltip on top" position="top">
            <Button variant="outline">Hover me (top)</Button>
          </Tooltip>
          <Tooltip content="This is a tooltip on bottom" position="bottom">
            <Button variant="outline">Hover me (bottom)</Button>
          </Tooltip>
          <Tooltip content="This is a tooltip on left" position="left">
            <Button variant="outline">Hover me (left)</Button>
          </Tooltip>
          <Tooltip content="This is a tooltip on right" position="right">
            <Button variant="outline">Hover me (right)</Button>
          </Tooltip>
        </div>
      </Card>

      {/* Modal */}
      <Card>
        <h2 className="text-xl font-semibold mb-4">Modal</h2>
        <Button onClick={() => setIsModalOpen(true)}>
          Open Modal
        </Button>
        <Modal
          isOpen={isModalOpen}
          onClose={() => setIsModalOpen(false)}
          title="Example Modal"
        >
          <div className="space-y-4">
            <p>This is a modal dialog example.</p>
            <div className="flex gap-2 justify-end">
              <Button
                variant="outline"
                onClick={() => setIsModalOpen(false)}
              >
                Cancel
              </Button>
              <Button onClick={() => setIsModalOpen(false)}>
                Confirm
              </Button>
            </div>
          </div>
        </Modal>
      </Card>

      {/* Toast Notifications */}
      <Card>
        <h2 className="text-xl font-semibold mb-4">Toast Notifications</h2>
        <div className="flex gap-2 flex-wrap">
          <Button
            variant="outline"
            onClick={() => showToast({
              type: 'success',
              title: 'Success!',
              message: 'Operation completed successfully.',
            })}
          >
            Success Toast
          </Button>
          <Button
            variant="outline"
            onClick={() => showToast({
              type: 'error',
              title: 'Error!',
              message: 'Something went wrong.',
            })}
          >
            Error Toast
          </Button>
          <Button
            variant="outline"
            onClick={() => showToast({
              type: 'warning',
              title: 'Warning!',
              message: 'Please check your input.',
            })}
          >
            Warning Toast
          </Button>
          <Button
            variant="outline"
            onClick={() => showToast({
              type: 'info',
              title: 'Info',
              message: 'Here is some information.',
            })}
          >
            Info Toast
          </Button>
        </div>
      </Card>

      {/* Empty States */}
      <Card>
        <h2 className="text-xl font-semibold mb-4">Empty States</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="border border-gray-200 rounded-lg">
            <EmptyState
              icon={Search}
              title="No results found"
              description="Try adjusting your search terms"
              action={{
                label: 'Clear filters',
                onClick: () => console.log('Clear filters'),
                variant: 'outline',
              }}
            />
          </div>
          <div className="border border-gray-200 rounded-lg">
            <EmptyState
              icon={Plus}
              title="No items yet"
              description="Get started by creating your first item"
              action={{
                label: 'Create item',
                onClick: () => console.log('Create item'),
              }}
            />
          </div>
        </div>
      </Card>

      {/* Form Component */}
      <Card>
        <h2 className="text-xl font-semibold mb-4">Form Component</h2>
        <Form
          validationSchema={sampleFormSchema}
          onSubmit={handleFormSubmit}
          className="space-y-4 max-w-md"
        >
          <InputField name="name" label="Name" placeholder="Enter your name" />
          <InputField
            name="email"
            type="email"
            label="Email"
            placeholder="Enter your email"
          />
          <TextareaField
            name="message"
            label="Message"
            placeholder="Enter your message"
          />
          <Button type="submit">Submit</Button>
        </Form>
      </Card>
    </div>
  );
};
