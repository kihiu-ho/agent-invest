import React from 'react';

const StatsCard = ({ title, value, icon: Icon, color = 'primary' }) => {
  const getColorClasses = () => {
    switch (color) {
      case 'success':
        return {
          bg: 'bg-success-50',
          icon: 'text-success-600',
          text: 'text-success-900'
        };
      case 'warning':
        return {
          bg: 'bg-warning-50',
          icon: 'text-warning-600',
          text: 'text-warning-900'
        };
      case 'error':
        return {
          bg: 'bg-error-50',
          icon: 'text-error-600',
          text: 'text-error-900'
        };
      default:
        return {
          bg: 'bg-primary-50',
          icon: 'text-primary-600',
          text: 'text-primary-900'
        };
    }
  };

  const colors = getColorClasses();

  return (
    <div className="card">
      <div className="flex items-center">
        <div className={`flex-shrink-0 p-3 rounded-lg ${colors.bg}`}>
          <Icon className={`w-6 h-6 ${colors.icon}`} />
        </div>
        <div className="ml-4">
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className={`text-2xl font-bold ${colors.text}`}>{value}</p>
        </div>
      </div>
    </div>
  );
};

export default StatsCard;
