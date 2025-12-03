# DQA Agent Guide

This guide explains how to use the AI-powered Data Quality Assessment (DQA) Agent to get insights about your sensor data quality.

## Overview

The DQA Agent is an AI-powered chat assistant that helps you:
- Ask questions about your sensor data quality
- Get insights and recommendations
- Understand data quality metrics
- Receive explanations of analysis results
- Get help with data quality issues

## Understanding the DQA Agent

### What is the DQA Agent?

The DQA Agent is a conversational AI assistant powered by OpenAI that:
- Understands your data quality context
- Accesses your sensor data and metadata
- Provides intelligent responses to questions
- Offers recommendations and insights
- Explains complex concepts in simple terms

### Capabilities

The agent can help with:
- **Data Quality Questions**: Ask about completeness, accuracy, consistency
- **Sensor Analysis**: Get insights about specific sensors
- **Metric Explanations**: Understand what quality metrics mean
- **Recommendations**: Receive suggestions for improving data quality
- **Troubleshooting**: Get help with data quality issues

## Accessing the DQA Agent

1. Open the application at http://localhost:8000
2. Click **DQA Agent** in the sidebar or home page
3. The chat interface opens automatically

## Using the DQA Agent

### Step 1: Select Machine Group (Optional)

1. Choose a machine group from the dropdown
2. This provides context for the agent
3. Agent can access data for selected machine group

**Note**: You can chat without selecting a machine group for general questions.

### Step 2: Start a Conversation

1. Type your question in the message input box
2. Press Enter or click Send
3. Wait for the agent's response

### Step 3: Continue Conversation

1. Read the agent's response
2. Ask follow-up questions
3. Request clarifications
4. Get more details on specific topics

## Types of Questions You Can Ask

### Data Quality Questions

**Examples:**
- "What is the overall data quality for this machine group?"
- "Which sensors have the most missing values?"
- "Are there any accuracy issues I should be concerned about?"
- "How does data quality compare across different sensors?"

### Sensor-Specific Questions

**Examples:**
- "Tell me about sensor 22PI102"
- "What are the threshold values for pressure sensors?"
- "Which sensors are most correlated?"
- "Are there any sensors with high outlier rates?"

### Metric Explanations

**Examples:**
- "What does completeness percentage mean?"
- "How is accuracy assessed?"
- "What are outliers and why do they matter?"
- "Explain correlation analysis"

### Recommendations

**Examples:**
- "What should I do about sensors with high missing values?"
- "How can I improve data quality?"
- "Which sensors need maintenance?"
- "What are best practices for data quality?"

### Troubleshooting

**Examples:**
- "Why is my data quality assessment showing poor results?"
- "What could cause high missing values?"
- "How do I fix accuracy issues?"
- "What should I check if sensors have many alarms?"

## Best Practices for Effective Queries

### Be Specific

**Good:**
- "What is the missing value percentage for sensor 22PI102?"
- "Which pressure sensors have accuracy issues?"

**Less Effective:**
- "Tell me about data"
- "What's wrong?"

### Provide Context

**Good:**
- "For the KT2201 machine group, which sensors have the most missing values?"
- "In the last month, what is the overall data quality?"

**Less Effective:**
- "What's the quality?"
- "Tell me about sensors"

### Ask Follow-Up Questions

**Example Conversation:**
1. User: "What is the data quality for KT2201?"
2. Agent: [Provides overview]
3. User: "Which specific sensors have issues?"
4. Agent: [Lists sensors with problems]
5. User: "What should I do about sensor 22PI102?"

### Use Natural Language

The agent understands natural language, so you can:
- Ask questions conversationally
- Use technical terms or plain language
- Request explanations
- Ask for recommendations

## Understanding Agent Responses

### Response Format

Agent responses may include:
- **Text Explanations**: Detailed answers to your questions
- **Lists**: Bulleted or numbered lists
- **Recommendations**: Actionable suggestions
- **Code/Data**: Technical details when relevant
- **Markdown Formatting**: Structured information

### Response Types

**Direct Answers:**
- Straightforward responses to specific questions
- Clear and concise information

**Analytical Responses:**
- Analysis of your data
- Comparisons and insights
- Pattern identification

**Recommendations:**
- Actionable suggestions
- Best practices
- Improvement strategies

**Explanations:**
- Concept explanations
- Metric definitions
- Process descriptions

## Limitations and Considerations

### Data Access

- Agent accesses data from selected machine group
- May not have access to all historical data
- Limited to data in the database

### Response Accuracy

- Responses are AI-generated and should be verified
- May not always be 100% accurate
- Use as guidance, not absolute truth

### Context Understanding

- Agent understands conversation context
- May need clarification for complex questions
- Works best with specific questions

### API Requirements

- Requires valid OpenAI API key
- Internet connection needed
- May have usage limits

## Tips for Better Results

1. **Select Machine Group**: Provides better context
2. **Be Specific**: Ask clear, specific questions
3. **Use Follow-Ups**: Build on previous responses
4. **Verify Information**: Cross-check important findings
5. **Ask for Examples**: Request examples when needed
6. **Request Clarification**: Ask agent to explain if unclear

## Example Conversations

### Example 1: Data Quality Overview

**User:** "What is the overall data quality for KT2201?"

**Agent:** "Based on the data for KT2201, the overall data quality is good. The completeness is 94.5%, which is above the acceptable threshold. However, there are some sensors with accuracy issues that should be addressed..."

**User:** "Which sensors have accuracy issues?"

**Agent:** "The sensors with the most accuracy issues are: 22PI102 (15% alarm rate), 22TI111 (12% alarm rate), and 22PI103 (10% alarm rate). These sensors are exceeding their threshold values frequently..."

### Example 2: Understanding Metrics

**User:** "What does completeness percentage mean?"

**Agent:** "Completeness percentage measures how much of the expected data is actually present. It's calculated as: (Actual Readings / Expected Readings) × 100. For example, if you expect 1000 readings but only have 950, the completeness is 95%..."

### Example 3: Getting Recommendations

**User:** "How can I improve data quality for my sensors?"

**Agent:** "Here are some recommendations to improve data quality: 1. Address sensors with high missing values by checking hardware and connections. 2. Calibrate sensors with accuracy issues. 3. Review and adjust threshold settings if needed. 4. Implement regular maintenance schedules..."

## Troubleshooting

### Agent Not Responding

**Problem**: No response from agent
**Solutions**:
- Check OpenAI API key is configured
- Verify internet connection
- Check backend service is running
- Review error messages

### Inaccurate Responses

**Problem**: Agent provides incorrect information
**Solutions**:
- Verify data is loaded correctly
- Check machine group selection
- Ask more specific questions
- Cross-check with other analyses

### Slow Responses

**Problem**: Agent takes too long to respond
**Solutions**:
- Wait for response (may take time)
- Check network connection
- Verify backend performance
- Reduce question complexity

## Next Steps

After using the DQA Agent:

1. **Verify Findings**: Cross-check with other analyses
2. **Take Action**: Implement recommendations
3. **Re-assess**: Check improvements after changes
4. **Explore More**: Use other analysis features

## Related Documentation

- [Data Quality Guide](data-quality-guide.md) - Comprehensive quality assessment
- [Missing Values Guide](missing-values-guide.md) - Missing data analysis
- [Invalid Values Guide](invalid-values-guide.md) - Invalid readings analysis

---

*Note: The DQA Agent requires a valid OpenAI API key. See [Getting Started Guide](getting-started.md) for setup instructions.*

