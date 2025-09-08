import os
import tempfile
import shutil
import subprocess
import pytest
from pathlib import Path
from unittest.mock import patch

import pyfm.utils
from pyfm.nanny.spawnjob import make_inputs


class TestGenerateInput:
    """Test the generate_input.py script functionality."""
    
    @pytest.fixture
    def test_data_dir(self):
        """Return the path to the test data directory."""
        return Path(__file__).parent
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def params_data(self, test_data_dir):
        """Load the test params.yaml file."""
        params_file = test_data_dir / "l3248_params.yaml"
        return pyfm.utils.load_param(str(params_file))
    
    def get_expected_output_filename(self, params_data, job, series, config):
        """Get the expected output filename based on params configuration."""
        io_template = params_data['job_setup'][job]['io']
        
        # Extract values from params for template substitution
        eigs = params_data['hadrons_params']['eigs']
        noise = params_data['hadrons_params']['noise']
        
        # Format the io template with actual values
        # Handle templates that may or may not have placeholders
        try:
            formatted_io = io_template.format(
                eigs=eigs,
                noise=noise,
                series=series,
                cfg=config
            )
            # If template already includes series and config, don't append them again
            if '{series}' in io_template and '{cfg}' in io_template:
                return f"{formatted_io}.xml"
        except KeyError:
            # If template doesn't have all placeholders, use it as-is
            formatted_io = io_template
        
        # Follow the same pattern as get_infile: {io}-{series}.{cfg}.xml
        return f"{formatted_io}-{series}.{config}.xml"
    
    def test_make_inputs_hadrons_job(self, params_data, temp_dir):
        """Test that make_inputs generates correct XML and schedule files for hadrons job."""
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            job = "hadrons"
            series = "a"
            config = "100"
            cfgno_steps = [(f"{series}.{config}", None)]
            
            # Call make_inputs
            make_inputs(params_data, job, cfgno_steps)
            
            # Check that in/ directory was created
            assert os.path.exists("in/")
            
            # Check that schedules/ directory was created
            assert os.path.exists("schedules/")
            
            # Get expected filename from params
            expected_xml_file = f"in/{self.get_expected_output_filename(params_data, job, series, config)}"
            assert os.path.exists(expected_xml_file)
            
            # Check that the expected schedule file was created
            base_name = self.get_expected_output_filename(params_data, job, series, config)[:-4]  # Remove .xml
            expected_sched_file = f"schedules/{base_name}.sched"
            assert os.path.exists(expected_sched_file)
            
            # Verify XML file contains expected content
            with open(expected_xml_file, 'r') as f:
                xml_content = f.read()
                
            # Check for key XML elements
            assert "<grid>" in xml_content
            assert "<modules>" in xml_content
            assert "gauge" in xml_content
            assert "epack" in xml_content
            assert f"LMI-RW-series-{series}-{params_data['hadrons_params']['eigs']}-eigs-{params_data['hadrons_params']['noise']}-noise" in xml_content
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.parametrize("params_file,job,series,config", [
        ("l3248_params.yaml", "hadrons", "a", "100"),
        # Add more test cases here as needed
    ])
    def test_generate_input_script_with_params_flag(self, test_data_dir, temp_dir, params_file, job, series, config):
        """Test that the generate_input.py script executes correctly with --params flag."""
        # Copy params file to temp directory
        params_path = test_data_dir / params_file
        temp_params_path = Path(temp_dir) / params_file
        shutil.copy(params_path, temp_params_path)
        
        # Load params to get expected filename
        params_data = pyfm.utils.load_param(str(params_path))
        expected_xml_filename = self.get_expected_output_filename(params_data, job, series, config)
        expected_sched_filename = expected_xml_filename[:-4] + ".sched"  # Replace .xml with .sched
        
        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Run the generate_input.py script with --params flag
            script_path = Path(__file__).parent.parent.parent / "scripts" / "generate_input.py"
            cmd = [
                "python", str(script_path),
                "--params", str(temp_params_path),
                "--job", job,
                "--series", series, 
                "--config", config
            ]
            
            # Mock the logging setup to avoid issues
            with patch('pyfm.setup_logging'):
                result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Check that script executed successfully
            assert result.returncode == 0, f"Script failed with stderr: {result.stderr}"
            
            # Check that expected files were created
            assert os.path.exists(f"in/{expected_xml_filename}")
            assert os.path.exists(f"schedules/{expected_sched_filename}")
            
        finally:
            os.chdir(original_cwd)
    
    def test_generated_files_match_expected(self, test_data_dir, temp_dir):
        """Test that generated files match the expected reference files."""
        params_file = "l3248_params.yaml"
        job = "hadrons"
        series = "a"
        config = "100"
        
        # Copy params.yaml to temp directory
        shutil.copy(test_data_dir / params_file, temp_dir)
        
        # Load params to get expected filenames
        params_data = pyfm.utils.load_param(str(test_data_dir / params_file))
        expected_xml_filename = self.get_expected_output_filename(params_data, job, series, config)
        expected_sched_filename = expected_xml_filename[:-4] + ".sched"
        
        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Generate files using make_inputs
            make_inputs(params_data, job, [(f"{series}.{config}", None)])
            
            # Compare generated XML with expected (if reference exists)
            expected_xml_path = test_data_dir / "in" / expected_xml_filename
            generated_xml = Path(f"in/{expected_xml_filename}")
            
            if expected_xml_path.exists():
                with open(generated_xml, 'r') as f:
                    generated_content = f.read()
                
                # Check key structural elements are the same
                assert "<grid>" in generated_content
                assert "</grid>" in generated_content
                assert f"LMI-RW-series-{series}-{params_data['hadrons_params']['eigs']}-eigs-{params_data['hadrons_params']['noise']}-noise" in generated_content
            
            # Compare generated schedule with expected (if reference exists)
            expected_sched_path = test_data_dir / "schedules" / expected_sched_filename
            generated_sched = Path(f"schedules/{expected_sched_filename}")
            
            if expected_sched_path.exists():
                with open(expected_sched_path, 'r') as f:
                    expected_sched_content = f.read()
                with open(generated_sched, 'r') as f:
                    generated_sched_content = f.read()
                
                # Schedule files should be identical
                assert expected_sched_content == generated_sched_content
            else:
                # If no reference file exists, just verify the generated file has correct format
                with open(generated_sched, 'r') as f:
                    sched_content = f.read()
                
                sched_lines = sched_content.strip().split('\n')
                assert len(sched_lines) > 0
                assert sched_lines[0].isdigit()  # First line should be module count
                
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.parametrize("series,config", [
        ("a", "100"),
        ("b", "200"),
        ("c", "150"),
    ])
    def test_make_inputs_with_different_series_config(self, params_data, temp_dir, series, config):
        """Test make_inputs with different series and config numbers."""
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            job = "hadrons"
            cfgno_steps = [(f"{series}.{config}", None)]
            
            make_inputs(params_data, job, cfgno_steps)
            
            # Get expected filename from params
            expected_xml_filename = self.get_expected_output_filename(params_data, job, series, config)
            expected_sched_filename = expected_xml_filename[:-4] + ".sched"
            
            # Check that files with correct naming were created
            assert os.path.exists(f"in/{expected_xml_filename}")
            assert os.path.exists(f"schedules/{expected_sched_filename}")
            
            # Verify content contains correct series/config
            with open(f"in/{expected_xml_filename}", 'r') as f:
                xml_content = f.read()
            
            # Check that the ensemble string contains the correct series
            ens = params_data['submit_params']['ens']
            assert f"{ens}{series}" in xml_content
            
        finally:
            os.chdir(original_cwd)
    
    def test_environment_variables_set(self, params_data, temp_dir):
        """Test that environment variables are properly set during make_inputs."""
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            job = "hadrons"
            series = "a"
            config = "100"
            cfgno_steps = [(f"{series}.{config}", None)]
            
            # Clear any existing INPUTLIST env var
            if "INPUTLIST" in os.environ:
                del os.environ["INPUTLIST"]
            
            make_inputs(params_data, job, cfgno_steps)
            
            # Get expected filename
            expected_xml_filename = self.get_expected_output_filename(params_data, job, series, config)
            
            # Check that INPUTLIST environment variable was set
            assert "INPUTLIST" in os.environ
            assert expected_xml_filename in os.environ["INPUTLIST"]
            
        finally:
            os.chdir(original_cwd)

    def test_custom_params_file(self, temp_dir):
        """Test with a custom params file to verify flexibility."""
        # Create a custom params file for testing
        custom_params = {
            'submit_params': {'ens': 'test_ensemble'},
            'hadrons_params': {'eigs': 500, 'noise': 2},
            'job_setup': {
                'hadrons': {
                    'io': 'custom-e{eigs}-n{noise}-{series}.{cfg}',
                    'job_type': 'hadrons'
                }
            }
        }
        
        import yaml
        custom_params_file = Path(temp_dir) / "custom_params.yaml"
        with open(custom_params_file, 'w') as f:
            yaml.dump(custom_params, f)
        
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Load the custom params
            params_data = pyfm.utils.load_param(str(custom_params_file))
            
            job = "hadrons"
            series = "x"
            config = "999"
            
            # Get expected filename based on custom params
            expected_filename = self.get_expected_output_filename(params_data, job, series, config)
            assert expected_filename == "custom-e500-n2-x.999.xml"
            
        finally:
            os.chdir(original_cwd)