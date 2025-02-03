# Adding New Requirements to Agent App

When you need to add new Python packages to the project, follow these steps:

1. **Update pyproject.toml**
   - Add your new package(s) to the `dependencies` list in `pyproject.toml`
   ```toml
   dependencies = [
     # ... existing dependencies ...
     "your-new-package"
   ]
   ```

2. **Generate Updated Requirements**
   - Run the following command to generate an updated `requirements.txt`:
   ```bash
   ./scripts/generate_requirements.sh upgrade
   ```

3. **Configure Build Settings**
   - Ensure `build_images=True` is set in `workspace/settings.py`
   - This setting tells phi to rebuild the Docker images with the new requirements

4. **Rebuild and Restart**
   - Run the following commands to stop the current containers, rebuild with new requirements, and start the updated containers:
   ```bash
   phi ws down
   phi ws up
   ```

## Important Notes

- Always update `pyproject.toml` first, don't modify `requirements.txt` directly
- The `requirements.txt` file is auto-generated and should not be edited manually
- When adding new requirements, make sure to activate your virtual environment first:
  ```bash
  source .venv/bin/activate
  ```
- If you encounter any issues, try removing the virtual environment and reinstalling:
  ```bash
  rm -rf .venv
  ./scripts/install.sh
  source .venv/bin/activate
  ```
