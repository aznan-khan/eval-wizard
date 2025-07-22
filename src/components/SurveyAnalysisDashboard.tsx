import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line } from 'recharts';
import { TrendingUp, Users, BookOpen, Star, MessageSquare, BarChart3 } from 'lucide-react';

const SurveyAnalysisDashboard = () => {
  const [analysisData, setAnalysisData] = useState(null);
  const [loading, setLoading] = useState(false);

  // Mock data structure based on the work plan
  const mockData = {
    overview: {
      totalResponses: 17,
      completionRate: 100,
      npsScore: 7.2,
      avgSatisfaction: 4.1
    },
    courseMetrics: [
      { course: 'AI Fundamentals', responses: 4, quality: 4.2, satisfaction: 4.3, nps: 8.1 },
      { course: 'Digital Marketing', responses: 5, quality: 3.9, satisfaction: 4.0, nps: 7.2 },
      { course: 'Content Creation', responses: 3, quality: 4.5, satisfaction: 4.4, nps: 8.5 },
      { course: 'App Development', responses: 3, quality: 4.0, satisfaction: 4.1, nps: 7.8 },
      { course: 'Financial Analysis', responses: 2, quality: 3.8, satisfaction: 3.9, nps: 6.5 }
    ],
    demographics: {
      gender: [
        { name: 'Female', value: 9, percentage: 53 },
        { name: 'Male', value: 7, percentage: 41 },
        { name: 'Other', value: 1, percentage: 6 }
      ],
      role: [
        { name: 'Student', value: 11, percentage: 65 },
        { name: 'Professional', value: 6, percentage: 35 }
      ]
    },
    insights: [
      {
        type: 'positive',
        title: 'High Content Quality',
        description: 'AI Fundamentals and Content Creation courses show exceptional content quality ratings (4.2+ out of 5)',
        priority: 'high'
      },
      {
        type: 'concern',
        title: 'Financial Analysis Improvement Needed',
        description: 'Lower satisfaction scores indicate need for course structure review',
        priority: 'high'
      },
      {
        type: 'opportunity',
        title: 'Mobile Feature Adoption',
        description: 'Students prefer mobile access but usage patterns show desktop preference',
        priority: 'medium'
      }
    ]
  };

  const runAnalysis = async () => {
    setLoading(true);
    // Simulate API call delay
    setTimeout(() => {
      setAnalysisData(mockData);
      setLoading(false);
    }, 2000);
  };

  const COLORS = ['hsl(var(--primary))', 'hsl(var(--secondary))', 'hsl(var(--accent))'];

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex justify-between items-start">
          <div>
            <h1 className="text-3xl font-bold text-foreground">Course Evaluation Analysis</h1>
            <p className="text-muted-foreground mt-2">Statistical insights with LLM-enhanced analysis</p>
          </div>
          <Button 
            onClick={runAnalysis} 
            disabled={loading}
            className="bg-primary text-primary-foreground hover:bg-primary/90"
          >
            {loading ? 'Analyzing...' : 'Run Analysis'}
            <BarChart3 className="ml-2 h-4 w-4" />
          </Button>
        </div>

        {loading && (
          <Card>
            <CardContent className="pt-6">
              <div className="space-y-4">
                <div className="flex items-center space-x-2">
                  <div className="h-2 w-2 bg-primary rounded-full animate-pulse"></div>
                  <span className="text-sm text-muted-foreground">Processing survey responses...</span>
                </div>
                <Progress value={33} className="w-full" />
                <div className="flex items-center space-x-2">
                  <div className="h-2 w-2 bg-primary rounded-full animate-pulse"></div>
                  <span className="text-sm text-muted-foreground">Running statistical analysis...</span>
                </div>
                <Progress value={66} className="w-full" />
                <div className="flex items-center space-x-2">
                  <div className="h-2 w-2 bg-primary rounded-full animate-pulse"></div>
                  <span className="text-sm text-muted-foreground">Generating LLM insights...</span>
                </div>
                <Progress value={90} className="w-full" />
              </div>
            </CardContent>
          </Card>
        )}

        {analysisData && (
          <>
            {/* Overview Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Total Responses</CardTitle>
                  <Users className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{analysisData.overview.totalResponses}</div>
                  <p className="text-xs text-muted-foreground">100% completion rate</p>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">NPS Score</CardTitle>
                  <TrendingUp className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{analysisData.overview.npsScore}</div>
                  <p className="text-xs text-muted-foreground">Above industry average</p>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Avg Satisfaction</CardTitle>
                  <Star className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{analysisData.overview.avgSatisfaction}/5</div>
                  <p className="text-xs text-muted-foreground">Strong performance</p>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Courses</CardTitle>
                  <BookOpen className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">5</div>
                  <p className="text-xs text-muted-foreground">Active programs</p>
                </CardContent>
              </Card>
            </div>

            {/* Main Analysis Tabs */}
            <Tabs defaultValue="overview" className="space-y-6">
              <TabsList className="grid w-full grid-cols-5">
                <TabsTrigger value="overview">Overview</TabsTrigger>
                <TabsTrigger value="courses">Course Analysis</TabsTrigger>
                <TabsTrigger value="demographics">Demographics</TabsTrigger>
                <TabsTrigger value="insights">AI Insights</TabsTrigger>
                <TabsTrigger value="recommendations">Actions</TabsTrigger>
              </TabsList>

              <TabsContent value="overview" className="space-y-6">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <Card>
                    <CardHeader>
                      <CardTitle>Course Performance Comparison</CardTitle>
                      <CardDescription>Quality and satisfaction scores by course</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={analysisData.courseMetrics}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="course" tick={{ fontSize: 12 }} />
                          <YAxis />
                          <Tooltip />
                          <Bar dataKey="quality" fill="hsl(var(--primary))" name="Quality" />
                          <Bar dataKey="satisfaction" fill="hsl(var(--secondary))" name="Satisfaction" />
                        </BarChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle>NPS Distribution</CardTitle>
                      <CardDescription>Net Promoter Score by course</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <ResponsiveContainer width="100%" height={300}>
                        <LineChart data={analysisData.courseMetrics}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="course" tick={{ fontSize: 12 }} />
                          <YAxis />
                          <Tooltip />
                          <Line type="monotone" dataKey="nps" stroke="hsl(var(--primary))" strokeWidth={2} />
                        </LineChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>
                </div>
              </TabsContent>

              <TabsContent value="courses" className="space-y-6">
                <div className="grid grid-cols-1 gap-6">
                  {analysisData.courseMetrics.map((course, index) => (
                    <Card key={index}>
                      <CardHeader>
                        <div className="flex justify-between items-start">
                          <div>
                            <CardTitle>{course.course}</CardTitle>
                            <CardDescription>{course.responses} responses</CardDescription>
                          </div>
                          <Badge variant={course.nps > 7.5 ? "default" : course.nps > 6.5 ? "secondary" : "destructive"}>
                            NPS: {course.nps}
                          </Badge>
                        </div>
                      </CardHeader>
                      <CardContent>
                        <div className="grid grid-cols-3 gap-4">
                          <div className="space-y-2">
                            <p className="text-sm font-medium">Quality Score</p>
                            <div className="flex items-center space-x-2">
                              <Progress value={course.quality * 20} className="flex-1" />
                              <span className="text-sm font-bold">{course.quality}/5</span>
                            </div>
                          </div>
                          <div className="space-y-2">
                            <p className="text-sm font-medium">Satisfaction</p>
                            <div className="flex items-center space-x-2">
                              <Progress value={course.satisfaction * 20} className="flex-1" />
                              <span className="text-sm font-bold">{course.satisfaction}/5</span>
                            </div>
                          </div>
                          <div className="space-y-2">
                            <p className="text-sm font-medium">Completion</p>
                            <div className="flex items-center space-x-2">
                              <Progress value={100} className="flex-1" />
                              <span className="text-sm font-bold">100%</span>
                            </div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </TabsContent>

              <TabsContent value="demographics" className="space-y-6">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <Card>
                    <CardHeader>
                      <CardTitle>Gender Distribution</CardTitle>
                      <CardDescription>Survey respondent demographics</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <ResponsiveContainer width="100%" height={300}>
                        <PieChart>
                          <Pie
                            data={analysisData.demographics.gender}
                            cx="50%"
                            cy="50%"
                            outerRadius={80}
                            fill="#8884d8"
                            dataKey="value"
                            label={({ name, percentage }) => `${name} ${percentage}%`}
                          >
                            {analysisData.demographics.gender.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                            ))}
                          </Pie>
                          <Tooltip />
                        </PieChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle>Role Distribution</CardTitle>
                      <CardDescription>Student vs Professional breakdown</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <ResponsiveContainer width="100%" height={300}>
                        <PieChart>
                          <Pie
                            data={analysisData.demographics.role}
                            cx="50%"
                            cy="50%"
                            outerRadius={80}
                            fill="#8884d8"
                            dataKey="value"
                            label={({ name, percentage }) => `${name} ${percentage}%`}
                          >
                            {analysisData.demographics.role.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                            ))}
                          </Pie>
                          <Tooltip />
                        </PieChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>
                </div>
              </TabsContent>

              <TabsContent value="insights" className="space-y-6">
                <div className="space-y-4">
                  <div className="flex items-center space-x-2">
                    <MessageSquare className="h-5 w-5 text-primary" />
                    <h3 className="text-lg font-semibold">LLM-Generated Insights</h3>
                  </div>
                  
                  {analysisData.insights.map((insight, index) => (
                    <Card key={index}>
                      <CardContent className="pt-6">
                        <div className="flex items-start space-x-4">
                          <Badge 
                            variant={insight.type === 'positive' ? 'default' : insight.type === 'concern' ? 'destructive' : 'secondary'}
                          >
                            {insight.type}
                          </Badge>
                          <div className="flex-1 space-y-2">
                            <h4 className="font-semibold">{insight.title}</h4>
                            <p className="text-muted-foreground">{insight.description}</p>
                            <Badge variant="outline">
                              Priority: {insight.priority}
                            </Badge>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </TabsContent>

              <TabsContent value="recommendations" className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Actionable Recommendations</CardTitle>
                    <CardDescription>Data-driven improvement suggestions</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-3">
                      <div className="flex items-center justify-between p-3 border rounded-lg">
                        <div>
                          <h4 className="font-medium">Enhance Financial Analysis Course</h4>
                          <p className="text-sm text-muted-foreground">Restructure content based on satisfaction feedback</p>
                        </div>
                        <Badge variant="destructive">High Priority</Badge>
                      </div>
                      
                      <div className="flex items-center justify-between p-3 border rounded-lg">
                        <div>
                          <h4 className="font-medium">Optimize Mobile Experience</h4>
                          <p className="text-sm text-muted-foreground">Bridge gap between preference and usage</p>
                        </div>
                        <Badge variant="secondary">Medium Priority</Badge>
                      </div>
                      
                      <div className="flex items-center justify-between p-3 border rounded-lg">
                        <div>
                          <h4 className="font-medium">Expand AI and Content Creation</h4>
                          <p className="text-sm text-muted-foreground">Leverage high-performing courses</p>
                        </div>
                        <Badge variant="default">Low Priority</Badge>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </>
        )}
      </div>
    </div>
  );
};

export default SurveyAnalysisDashboard;