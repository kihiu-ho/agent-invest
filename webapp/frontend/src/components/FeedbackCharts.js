import React, { useEffect, useRef } from 'react';
import { Chart as ChartJS, registerables } from 'chart.js';
import 'chartjs-adapter-date-fns';

// Register Chart.js components
ChartJS.register(...registerables);

const FeedbackCharts = ({ trendsData, overviewData, reportsData }) => {
  const trendsChartRef = useRef(null);
  const distributionChartRef = useRef(null);
  const reportsChartRef = useRef(null);
  const trendsChartInstance = useRef(null);
  const distributionChartInstance = useRef(null);
  const reportsChartInstance = useRef(null);

  // Cleanup function to destroy existing charts
  const destroyChart = (chartInstance) => {
    if (chartInstance.current) {
      chartInstance.current.destroy();
      chartInstance.current = null;
    }
  };

  // Create trends chart
  useEffect(() => {
    if (!trendsData || !trendsChartRef.current) return;

    destroyChart(trendsChartInstance);

    const ctx = trendsChartRef.current.getContext('2d');
    
    trendsChartInstance.current = new ChartJS(ctx, {
      type: 'line',
      data: {
        labels: trendsData.daily_stats.map(stat => stat.date),
        datasets: [
          {
            label: 'Thumbs Up',
            data: trendsData.daily_stats.map(stat => stat.thumbs_up),
            borderColor: 'rgb(34, 197, 94)',
            backgroundColor: 'rgba(34, 197, 94, 0.1)',
            tension: 0.4,
            fill: true
          },
          {
            label: 'Thumbs Down',
            data: trendsData.daily_stats.map(stat => stat.thumbs_down),
            borderColor: 'rgb(239, 68, 68)',
            backgroundColor: 'rgba(239, 68, 68, 0.1)',
            tension: 0.4,
            fill: true
          },
          {
            label: 'Total Feedback',
            data: trendsData.daily_stats.map(stat => stat.total),
            borderColor: 'rgb(59, 130, 246)',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            tension: 0.4,
            fill: false,
            borderDash: [5, 5]
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: true,
            text: `Feedback Trends (Last ${trendsData.period_days} Days)`,
            font: {
              size: 16,
              weight: 'bold'
            }
          },
          legend: {
            position: 'top',
          }
        },
        scales: {
          x: {
            type: 'time',
            time: {
              unit: 'day',
              displayFormats: {
                day: 'MMM dd'
              }
            },
            title: {
              display: true,
              text: 'Date'
            }
          },
          y: {
            beginAtZero: true,
            ticks: {
              stepSize: 1
            },
            title: {
              display: true,
              text: 'Feedback Count'
            }
          }
        },
        interaction: {
          intersect: false,
          mode: 'index'
        }
      }
    });

    return () => destroyChart(trendsChartInstance);
  }, [trendsData]);

  // Create distribution pie chart
  useEffect(() => {
    if (!overviewData || !distributionChartRef.current) return;

    destroyChart(distributionChartInstance);

    const ctx = distributionChartRef.current.getContext('2d');
    
    distributionChartInstance.current = new ChartJS(ctx, {
      type: 'doughnut',
      data: {
        labels: ['Thumbs Up', 'Thumbs Down'],
        datasets: [{
          data: [overviewData.thumbs_up_count, overviewData.thumbs_down_count],
          backgroundColor: [
            'rgb(34, 197, 94)',
            'rgb(239, 68, 68)'
          ],
          borderColor: [
            'rgb(34, 197, 94)',
            'rgb(239, 68, 68)'
          ],
          borderWidth: 2
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: true,
            text: 'Feedback Distribution',
            font: {
              size: 16,
              weight: 'bold'
            }
          },
          legend: {
            position: 'bottom',
          }
        }
      }
    });

    return () => destroyChart(distributionChartInstance);
  }, [overviewData]);

  // Create reports bar chart
  useEffect(() => {
    if (!reportsData || !reportsChartRef.current) return;

    destroyChart(reportsChartInstance);

    const ctx = reportsChartRef.current.getContext('2d');
    
    reportsChartInstance.current = new ChartJS(ctx, {
      type: 'bar',
      data: {
        labels: reportsData.reports.map(report => 
          report.report_id.substring(0, 8) + '...'
        ),
        datasets: [
          {
            label: 'Thumbs Up',
            data: reportsData.reports.map(report => report.thumbs_up),
            backgroundColor: 'rgba(34, 197, 94, 0.8)',
            borderColor: 'rgb(34, 197, 94)',
            borderWidth: 1
          },
          {
            label: 'Thumbs Down',
            data: reportsData.reports.map(report => report.thumbs_down),
            backgroundColor: 'rgba(239, 68, 68, 0.8)',
            borderColor: 'rgb(239, 68, 68)',
            borderWidth: 1
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: true,
            text: 'Feedback by Report (Top 10)',
            font: {
              size: 16,
              weight: 'bold'
            }
          },
          legend: {
            position: 'top',
          }
        },
        scales: {
          x: {
            stacked: true,
          },
          y: {
            stacked: true,
            beginAtZero: true,
            ticks: {
              stepSize: 1
            }
          }
        }
      }
    });

    return () => destroyChart(reportsChartInstance);
  }, [reportsData]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
      {/* Trends Chart */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="h-80">
          <canvas ref={trendsChartRef}></canvas>
        </div>
      </div>

      {/* Distribution Chart */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="h-80">
          <canvas ref={distributionChartRef}></canvas>
        </div>
      </div>

      {/* Reports Chart - Full Width */}
      <div className="lg:col-span-2 bg-white rounded-lg shadow p-6">
        <div className="h-80">
          <canvas ref={reportsChartRef}></canvas>
        </div>
      </div>
    </div>
  );
};

export default FeedbackCharts;
