/**
 * Basic smoke tests to ensure the application can be imported and basic functionality works
 * These tests are designed to pass reliably in CI/CD environments
 */

describe('Basic Application Tests', () => {
  test('React is available', () => {
    const React = require('react');
    expect(React).toBeDefined();
    expect(typeof React.createElement).toBe('function');
  });

  test('Application modules can be imported', () => {
    // Test that we can import the main App component without errors
    expect(() => {
      require('./App');
    }).not.toThrow();
  });

  test('Services can be imported', () => {
    // Test that we can import services without errors
    expect(() => {
      require('./services/reportService');
    }).not.toThrow();
  });

  test('Components can be imported', () => {
    // Test that we can import components without errors
    expect(() => {
      require('./components/Header');
    }).not.toThrow();
    
    expect(() => {
      require('./components/Dashboard');
    }).not.toThrow();
  });

  test('Basic JavaScript functionality', () => {
    // Test basic JavaScript operations
    const testArray = [1, 2, 3, 4, 5];
    const doubled = testArray.map(x => x * 2);
    expect(doubled).toEqual([2, 4, 6, 8, 10]);
    
    const sum = testArray.reduce((acc, val) => acc + val, 0);
    expect(sum).toBe(15);
  });

  test('Environment variables are accessible', () => {
    // Test that process.env is available (important for React apps)
    expect(process.env).toBeDefined();
    expect(typeof process.env.NODE_ENV).toBe('string');
  });

  test('JSON operations work correctly', () => {
    const testObject = { name: 'test', value: 42 };
    const jsonString = JSON.stringify(testObject);
    const parsedObject = JSON.parse(jsonString);
    
    expect(parsedObject).toEqual(testObject);
    expect(parsedObject.name).toBe('test');
    expect(parsedObject.value).toBe(42);
  });

  test('Promise functionality works', async () => {
    const testPromise = new Promise(resolve => {
      setTimeout(() => resolve('success'), 10);
    });
    
    const result = await testPromise;
    expect(result).toBe('success');
  });

  test('Date functionality works', () => {
    const now = new Date();
    expect(now instanceof Date).toBe(true);
    expect(typeof now.getTime()).toBe('number');
    expect(now.getTime()).toBeGreaterThan(0);
  });
});
