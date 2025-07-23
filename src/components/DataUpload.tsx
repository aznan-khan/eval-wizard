import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useToast } from "@/hooks/use-toast";
import { Upload, FileText } from 'lucide-react';

interface DataUploadProps {
  onUploadSuccess: () => void;
}

const DataUpload: React.FC<DataUploadProps> = ({ onUploadSuccess }) => {
  const [uploading, setUploading] = useState(false);
  const { toast } = useToast();

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploading(true);
    
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('http://localhost:8000/api/survey/responses/upload', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        toast({
          title: "Upload Successful",
          description: `${result.stored_responses} responses uploaded successfully.`,
        });
        onUploadSuccess();
      } else {
        throw new Error('Upload failed');
      }
    } catch (error) {
      toast({
        title: "Upload Failed",
        description: "Could not upload the file. Please check your data format.",
        variant: "destructive",
      });
    } finally {
      setUploading(false);
    }
  };

  const generateMockData = async () => {
    setUploading(true);
    
    try {
      const response = await fetch('http://localhost:8000/api/survey/responses/generate-mock?num_responses=17', {
        method: 'POST'
      });

      if (response.ok) {
        const result = await response.json();
        toast({
          title: "Mock Data Generated",
          description: `${result.generated_responses} mock responses created successfully.`,
        });
        onUploadSuccess();
      } else {
        throw new Error('Mock data generation failed');
      }
    } catch (error) {
      toast({
        title: "Generation Failed",
        description: "Could not generate mock data.",
        variant: "destructive",
      });
    } finally {
      setUploading(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center">
          <FileText className="mr-2 h-5 w-5" />
          Data Management
        </CardTitle>
        <CardDescription>
          Upload survey responses or generate mock data for analysis
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <label htmlFor="file-upload" className="text-sm font-medium">
            Upload Survey Data (JSON)
          </label>
          <Input
            id="file-upload"
            type="file"
            accept=".json"
            onChange={handleFileUpload}
            disabled={uploading}
          />
        </div>
        
        <div className="flex justify-between items-center pt-4 border-t">
          <Button
            variant="outline"
            onClick={generateMockData}
            disabled={uploading}
          >
            Generate Mock Data
          </Button>
          
          <div className="text-sm text-muted-foreground">
            {uploading ? 'Processing...' : 'Ready to upload'}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default DataUpload;