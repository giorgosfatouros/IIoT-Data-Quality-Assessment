# Troubleshooting Guide

This guide helps you resolve common issues when using the IIoT Data Quality Assessment Service.

## Table of Contents

- [Service Connection Problems](#service-connection-problems)
- [Data Loading Issues](#data-loading-issues)
- [Import Failures](#import-failures)
- [Performance Issues](#performance-issues)
- [Browser Compatibility](#browser-compatibility)
- [Getting Help](#getting-help)

## Service Connection Problems

### Frontend Not Loading

**Symptoms**: Cannot access http://localhost:5173

**Solutions**:
1. Check if frontend service is running:
   ```bash
   # Check if process is running
   lsof -ti tcp:5173
   ```

2. Restart frontend:
   ```bash
   cd frontend
   npm run dev
   ```

3. Check for port conflicts:
   ```bash
   # Kill process on port 5173 if needed
   lsof -ti tcp:5173 | xargs kill -9
   ```

4. Verify Node.js is installed:
   ```bash
   node --version  # Should be 18+
   ```

### Backend API Not Responding

**Symptoms**: API calls fail, 500 errors, or connection refused

**Solutions**:
1. Check if backend service is running:
   ```bash
   docker-compose ps backend
   ```

2. Check backend logs:
   ```bash
   docker-compose logs backend
   ```

3. Verify backend health:
   ```bash
   curl http://localhost:8000/health
   ```

4. Restart backend service:
   ```bash
   docker-compose restart backend
   ```

5. Check for port conflicts:
   ```bash
   lsof -ti tcp:8000 | xargs kill -9
   ```

### Database Connection Issues

**Symptoms**: Database errors, connection timeouts

**Solutions**:
1. Check if TimescaleDB is running:
   ```bash
   docker-compose ps timescaledb
   ```

2. Check database logs:
   ```bash
   docker-compose logs timescaledb
   ```

3. Verify database connection:
   ```bash
   docker-compose exec timescaledb pg_isready -U iiot_user -d iiot_dqa
   ```

4. Check environment variables:
   ```bash
   cat .env | grep DB_
   ```

5. Restart database:
   ```bash
   docker-compose restart timescaledb
   ```

## Data Loading Issues

### No Tables Available

**Symptoms**: Dropdown is empty, no machine groups shown

**Solutions**:
1. Import data first (see [Data Import Guide](data-import-guide.md))
2. Verify data was imported successfully
3. Check database for tables:
   ```bash
   docker-compose exec timescaledb psql -U iiot_user -d iiot_dqa -c "\dt"
   ```
4. Refresh the page
5. Check backend API:
   ```bash
   curl http://localhost:8000/tables
   ```

### Data Not Loading

**Symptoms**: Selected table but no data appears

**Solutions**:
1. Check backend logs for errors:
   ```bash
   docker-compose logs backend | tail -50
   ```
2. Verify table has data:
   ```bash
   docker-compose exec timescaledb psql -U iiot_user -d iiot_dqa -c "SELECT COUNT(*) FROM your_table_name;"
   ```
3. Check browser console for errors (F12)
4. Verify API endpoint:
   ```bash
   curl "http://localhost:8000/data?table=YOUR_TABLE&limit=10"
   ```
5. Refresh the page
6. Try selecting table again

### Missing Sensors

**Symptoms**: Expected sensors not visible

**Solutions**:
1. Verify sensors were imported (check import logs)
2. Check sensor tags match metadata
3. Review data import process
4. Verify sensor columns exist in data:
   ```bash
   docker-compose exec timescaledb psql -U iiot_user -d iiot_dqa -c "\d your_table_name"
   ```

## Import Failures

### Validation Errors

**Symptoms**: Files fail validation

**Solutions**:
1. Check CSV format:
   - Ensure proper CSV encoding (UTF-8)
   - Verify no extra commas or quotes
   - Check column names match requirements

2. Verify required columns:
   - Data file must have `timestamp` column
   - Tags file must have all required columns
   - See [Data Import Guide](data-import-guide.md) for format

3. Check file encoding:
   ```bash
   file -i your_file.csv  # Should show UTF-8
   ```

4. Review validation error messages
5. Fix issues and re-validate

### Import Job Fails

**Symptoms**: Import job status shows "failed"

**Solutions**:
1. Check import job status:
   ```bash
   curl "http://localhost:8000/import/status/JOB_ID"
   ```

2. Check backend logs:
   ```bash
   docker-compose logs backend | grep -i import
   ```

3. Verify database connection:
   ```bash
   docker-compose exec timescaledb pg_isready -U iiot_user -d iiot_dqa
   ```

4. Check disk space:
   ```bash
   df -h
   ```

5. Verify file sizes (not too large)
6. Try importing smaller subset first

### Slow Import

**Symptoms**: Import takes very long time

**Solutions**:
1. Large files take time - be patient
2. Check database performance
3. Monitor import progress
4. Consider importing subset of sensors first
5. Verify worker service is running:
   ```bash
   docker-compose ps worker
   ```

## Performance Issues

### Slow Page Loading

**Symptoms**: Pages load slowly

**Solutions**:
1. Check network connection
2. Reduce date range in filters
3. Select fewer sensors
4. Check browser performance (close other tabs)
5. Clear browser cache
6. Check backend performance:
   ```bash
   docker-compose stats
   ```

### Slow Chart Rendering

**Symptoms**: Charts take long to render

**Solutions**:
1. Use date range filters to reduce data
2. Select fewer sensors
3. Use aggregated data source
4. Check data volume
5. Wait for charts to fully load

### High Memory Usage

**Symptoms**: Browser or system becomes slow

**Solutions**:
1. Close unnecessary browser tabs
2. Reduce data range
3. Select fewer sensors
4. Restart browser
5. Check system resources:
   ```bash
   free -h
   top
   ```

## Browser Compatibility

### Browser Not Supported

**Symptoms**: Features don't work, layout broken

**Solutions**:
1. Use supported browsers:
   - Chrome/Edge (latest)
   - Firefox (latest)
   - Safari (latest)

2. Update browser to latest version
3. Clear browser cache
4. Disable browser extensions
5. Try incognito/private mode

### JavaScript Errors

**Symptoms**: Console shows errors, features broken

**Solutions**:
1. Open browser console (F12)
2. Check error messages
3. Clear browser cache
4. Disable browser extensions
5. Try different browser
6. Check browser console for specific errors

## Getting Help

### Check Logs

**Backend Logs**:
```bash
docker-compose logs backend
docker-compose logs backend --tail=100
```

**Frontend Logs**:
- Open browser console (F12)
- Check Console tab for errors
- Check Network tab for failed requests

**Database Logs**:
```bash
docker-compose logs timescaledb
```

### Verify Services

**Check all services are running**:
```bash
docker-compose ps
```

**Check service health**:
```bash
# Backend health
curl http://localhost:8000/health

# Database health
docker-compose exec timescaledb pg_isready -U iiot_user -d iiot_dqa
```

### Common Error Messages

**"Connection refused"**:
- Service not running
- Wrong port
- Firewall blocking

**"404 Not Found"**:
- Wrong URL
- Service not started
- Route doesn't exist

**"500 Internal Server Error"**:
- Backend error
- Check backend logs
- Database connection issue

**"CORS error"**:
- Backend CORS configuration
- Check backend settings
- Verify API URL

### Environment Issues

**Check environment variables**:
```bash
cat .env
```

**Verify configuration**:
- Database credentials correct
- OpenAI API key set (if using DQA Agent)
- Ports not in use

### Reset Services

**Restart all services**:
```bash
docker-compose down
docker-compose up -d
```

**Rebuild services** (if needed):
```bash
docker-compose down
docker-compose up --build -d
```

## Prevention Tips

1. **Regular Backups**: Backup data regularly
2. **Monitor Logs**: Check logs periodically
3. **Update Services**: Keep services updated
4. **Monitor Resources**: Check disk space and memory
5. **Test After Changes**: Verify after configuration changes

## Related Documentation

- [Getting Started Guide](getting-started.md) - Initial setup
- [Data Import Guide](data-import-guide.md) - Import troubleshooting
- [FAQ](faq.md) - Common questions

---

*For technical support, check the main [README](../README.md) or project documentation.*

