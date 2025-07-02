"""
Task 2: Automated Testing with AI
Login page automation testing using Selenium with AI-enhanced capabilities
"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple

class AIEnhancedLoginTester:
    """
    AI-enhanced login testing class that simulates intelligent test case generation
    and execution with adaptive element detection
    """
    
    def __init__(self, headless: bool = True):
        """Initialize the tester with Chrome driver"""
        self.options = Options()
        if headless:
            self.options.add_argument("--headless")
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")
        self.driver = None
        self.test_results = []
        
    def setup_driver(self):
        """Setup Chrome WebDriver"""
        try:
            self.driver = webdriver.Chrome(options=self.options)
            self.driver.implicitly_wait(10)
            return True
        except Exception as e:
            print(f"Failed to setup driver: {e}")
            return False
    
    def ai_element_detection(self, page_source: str) -> Dict[str, str]:
        """
        AI-simulated intelligent element detection
        In real AI testing tools, this would use ML to identify form elements
        """
        # Simulate AI detection of form elements based on common patterns
        selectors = {}
        
        # Common username field patterns
        username_patterns = [
            "input[name*='user']",
            "input[id*='user']", 
            "input[placeholder*='username']",
            "input[type='email']",
            "#username", "#user", "#email"
        ]
        
        # Common password field patterns  
        password_patterns = [
            "input[type='password']",
            "input[name*='pass']",
            "input[id*='pass']",
            "#password", "#pass"
        ]
        
        # Common submit button patterns
        submit_patterns = [
            "button[type='submit']",
            "input[type='submit']",
            "button:contains('Login')",
            "button:contains('Sign in')",
            ".login-btn", "#login-btn"
        ]
        
        selectors['username'] = username_patterns
        selectors['password'] = password_patterns  
        selectors['submit'] = submit_patterns
        
        return selectors
    
    def smart_element_finder(self, element_type: str) -> object:
        """
        AI-enhanced element finding with multiple fallback strategies
        """
        selectors = self.ai_element_detection(self.driver.page_source)
        
        for selector in selectors.get(element_type, []):
            try:
                element = self.driver.find_element(By.CSS_SELECTOR, selector)
                if element.is_displayed():
                    return element
            except NoSuchElementException:
                continue
                
        # Fallback to xpath and text-based search
        if element_type == 'submit':
            try:
                return self.driver.find_element(By.XPATH, "//button[contains(text(), 'Login') or contains(text(), 'Sign in')]")
            except NoSuchElementException:
                pass
                
        return None
    
    def create_test_login_page(self):
        """Create a simple HTML login page for testing"""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Login Page</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 50px; }
                .login-form { max-width: 400px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; }
                input { width: 100%; padding: 10px; margin: 10px 0; box-sizing: border-box; }
                button { width: 100%; padding: 10px; background: #007bff; color: white; border: none; cursor: pointer; }
                .error { color: red; margin-top: 10px; }
            </style>
        </head>
        <body>
            <div class="login-form">
                <h2>Login</h2>
                <form id="loginForm">
                    <input type="text" id="username" placeholder="Username" required>
                    <input type="password" id="password" placeholder="Password" required>
                    <button type="submit" id="loginBtn">Login</button>
                </form>
                <div id="message" class="error"></div>
            </div>
            
            <script>
                document.getElementById('loginForm').addEventListener('submit', function(e) {
                    e.preventDefault();
                    const username = document.getElementById('username').value;
                    const password = document.getElementById('password').value;
                    const message = document.getElementById('message');
                    
                    if (username === 'admin' && password === 'password123') {
                        message.style.color = 'green';
                        message.textContent = 'Login successful!';
                    } else {
                        message.style.color = 'red';
                        message.textContent = 'Invalid credentials!';
                    }
                });
            </script>
        </body>
        </html>
        """
        
        # Save HTML file
        with open('/tmp/test_login.html', 'w') as f:
            f.write(html_content)
        
        return 'file:///tmp/test_login.html'
    
    def run_test_case(self, test_name: str, username: str, password: str, expected_result: str) -> Dict:
        """
        Run a single test case with AI-enhanced error detection
        """
        start_time = time.time()
        test_result = {
            'test_name': test_name,
            'username': username,
            'password': password,
            'expected_result': expected_result,
            'actual_result': 'FAILED',
            'execution_time': 0,
            'timestamp': datetime.now().isoformat(),
            'error_details': None
        }
        
        try:
            # Navigate to login page
            url = self.create_test_login_page()
            self.driver.get(url)
            
            # AI-enhanced element detection
            username_field = self.smart_element_finder('username')
            password_field = self.smart_element_finder('password')
            submit_button = self.smart_element_finder('submit')
            
            if not all([username_field, password_field, submit_button]):
                test_result['error_details'] = "Could not locate all required form elements"
                return test_result
            
            # Clear and input credentials
            username_field.clear()
            username_field.send_keys(username)
            
            password_field.clear() 
            password_field.send_keys(password)
            
            # Submit form
            submit_button.click()
            
            # Wait for response and check result
            time.sleep(2)
            
            try:
                message_element = self.driver.find_element(By.ID, "message")
                message_text = message_element.text
                
                if expected_result == 'success' and 'successful' in message_text.lower():
                    test_result['actual_result'] = 'PASSED'
                elif expected_result == 'failure' and 'invalid' in message_text.lower():
                    test_result['actual_result'] = 'PASSED'
                else:
                    test_result['actual_result'] = 'FAILED'
                    test_result['error_details'] = f"Unexpected message: {message_text}"
                    
            except NoSuchElementException:
                test_result['error_details'] = "Could not find result message element"
            
        except Exception as e:
            test_result['error_details'] = str(e)
        
        test_result['execution_time'] = time.time() - start_time
        self.test_results.append(test_result)
        return test_result
    
    def run_comprehensive_test_suite(self) -> List[Dict]:
        """
        Run comprehensive test suite with AI-generated test cases
        """
        # AI-simulated test case generation
        test_cases = [
            ("Valid Login", "admin", "password123", "success"),
            ("Invalid Username", "wronguser", "password123", "failure"), 
            ("Invalid Password", "admin", "wrongpass", "failure"),
            ("Empty Username", "", "password123", "failure"),
            ("Empty Password", "admin", "", "failure"),
            ("Both Empty", "", "", "failure"),
            ("SQL Injection Attempt", "admin'; DROP TABLE users; --", "password", "failure"),
            ("XSS Attempt", "<script>alert('xss')</script>", "password", "failure"),
            ("Special Characters", "user@domain.com", "p@ssw0rd!", "failure"),
            ("Long Input", "a" * 100, "b" * 100, "failure")
        ]
        
        print("Starting AI-Enhanced Login Test Suite...")
        print("=" * 60)
        
        for test_name, username, password, expected in test_cases:
            print(f"Running: {test_name}")
            result = self.run_test_case(test_name, username, password, expected)
            status = "✓ PASS" if result['actual_result'] == 'PASSED' else "✗ FAIL"
            print(f"  Result: {status} ({result['execution_time']:.2f}s)")
            if result['error_details']:
                print(f"  Error: {result['error_details']}")
            print()
        
        return self.test_results
    
    def generate_test_report(self) -> Dict:
        """Generate comprehensive test report with AI insights"""
        if not self.test_results:
            return {}
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r['actual_result'] == 'PASSED'])
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests) * 100
        
        avg_execution_time = sum(r['execution_time'] for r in self.test_results) / total_tests
        
        report = {
            'summary': {
                'total_tests': total_tests,
                'passed': passed_tests, 
                'failed': failed_tests,
                'success_rate': success_rate,
                'avg_execution_time': avg_execution_time
            },
            'detailed_results': self.test_results,
            'ai_insights': {
                'coverage_analysis': 'Tests cover authentication, input validation, and security scenarios',
                'risk_assessment': 'High risk areas: SQL injection, XSS, boundary conditions',
                'recommendations': [
                    'Implement rate limiting for login attempts',
                    'Add CAPTCHA for suspicious activity', 
                    'Strengthen input validation',
                    'Monitor for unusual login patterns'
                ]
            }
        }
        
        return report
    
    def save_results(self, filename: str = None):
        """Save test results to JSON file"""
        if not filename:
            filename = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = self.generate_test_report()
        
        filepath = f"/home/amirul/Desktop/Career/Class Academy/Specialization/AI/Assignment Four AI SW Engineering/task2_automated_testing/{filename}"
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Test results saved to: {filepath}")
        return filepath
    
    def cleanup(self):
        """Clean up resources"""
        if self.driver:
            self.driver.quit()

def main():
    """Main execution function"""
    tester = AIEnhancedLoginTester(headless=True)
    
    try:
        if not tester.setup_driver():
            print("Failed to setup WebDriver")
            return
        
        # Run comprehensive test suite
        results = tester.run_comprehensive_test_suite()
        
        # Generate and display report
        report = tester.generate_test_report()
        
        print("=" * 60)
        print("TEST EXECUTION SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {report['summary']['total_tests']}")
        print(f"Passed: {report['summary']['passed']}")
        print(f"Failed: {report['summary']['failed']}")
        print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
        print(f"Average Execution Time: {report['summary']['avg_execution_time']:.2f}s")
        
        print("\nAI INSIGHTS:")
        for insight in report['ai_insights']['recommendations']:
            print(f"  • {insight}")
        
        # Save results
        tester.save_results()
        
    finally:
        tester.cleanup()

if __name__ == "__main__":
    main()
