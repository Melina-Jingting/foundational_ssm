import subprocess
import re
import concurrent.futures
import time

#  "whitebait", "vimba" 
gpu_3090 = [
    "eider", "gressingham", "cackling", "shelduck","pochard" "bufflehead" "cackling",
    "gadwall", "pochard", 
    "aylesbury", "barnacle", 
    "eider", "gadwall", "pintail",
    "mallard", "mandarin", "pintail", "ruddy",
    "scaup", "shelduck", "shoveler", "smew", 
    "brent", "goosander", "gressingham", "wigeon",  "barbury", "crested", "harlequin", #"scoter"
    ]
    
gpu_4070 = [
    "albacore", "barbel", "chub", "dory", "elver", "goldeye",
    "hake", "inanga", "javelin", "koi", "lamprey", "mackerel", "mullet",
    "nase", "opah", "pike", "plaice", "quillback", "roach", "rudd",
    "shark", "skate", "tope", "uaru", "yellowtail", "zander"
    ] # "flounder"


def get_availability(host, timeout=20):    
    print(f"Checking {host}...")
    try:
        result = subprocess.run(
            ['ssh',
             '-o', 'ServerAliveInterval=5',
             '-o', 'ServerAliveCountMax=1',
             host, 'nvidia-smi'
        ],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        # Extract GPU name information using regex
        name_pattern = r'\|\s+\d+\s+(NVIDIA\s+\S+\s+\S+\s+\S+)'
        name_matches = re.findall(name_pattern, result.stdout)
        
        # Extract memory information using regex
        memory_pattern = r'(\d+)MiB\s+/\s+(\d+)MiB'
        memory_matches = re.findall(memory_pattern, result.stdout)
        
        # Extract power information using regex
        power_pattern = r'(\d+)W\s+/\s+(\d+)W'
        power_matches = re.findall(power_pattern, result.stdout)
        
        gpu_info = []
        if memory_matches and power_matches:
            for i, (memory_match, power_match) in enumerate(zip(memory_matches, power_matches)):
                gpu_name = name_matches[i].strip() if i < len(name_matches) else "Unknown GPU"
                used_memory, total_memory = memory_match
                used_power, total_power = power_match
                gpu_info.append({
                    'gpu': i,
                    'host': host,
                    'name': gpu_name,
                    'used_memory': int(used_memory),
                    'total_memory': int(total_memory),
                    'used_power': int(used_power),
                    'total_power': int(total_power),
                    'status': 'online'
                })
        
        # Print any errors if they occurred
        if result.stderr:
            print(f"Errors from {host}:")
            print(result.stderr)
        
        return gpu_info
    
    except subprocess.TimeoutExpired:
        print(f"Timeout connecting to {host} (exceeded {timeout} seconds)")
        return [{'host': host, 'status': 'timeout', 'error': f'Connection timed out after {timeout}s'}]
    
    except subprocess.CalledProcessError as e:
        print(f"Error connecting to {host}: {e}")
        return [{'host': host, 'status': 'error', 'error': str(e)}]
    
    except Exception as e:
        print(f"Unexpected error with {host}: {e}")
        return [{'host': host, 'status': 'error', 'error': str(e)}]

def print_gpu_info(gpu_info_list):
    online_hosts = []
    offline_hosts = []
    
    for host_gpus in gpu_info_list:
        if not host_gpus:  # Skip if no GPU info was returned
            continue
            
        # Check if this is an error entry
        if 'status' in host_gpus[0] and host_gpus[0]['status'] in ['error', 'timeout']:
            host = host_gpus[0]['host']
            error = host_gpus[0].get('error', 'Unknown error')
            offline_hosts.append((host, error))
            continue
            
        host = host_gpus[0]['host']  # Get host from first GPU
        online_hosts.append(host)
        print(f"\n--- {host} ---")
        for info in host_gpus:
            print(f"GPU {info['gpu']} ({info['name']})")
            print(f"  Memory: {info['used_memory']}MiB used / {info['total_memory']}MiB total")
            print(f"  Power: {info['used_power']}W used / {info['total_power']}W total")
    
    # Print summary of offline hosts
    if offline_hosts:
        print("\n=== OFFLINE HOSTS ===")
        for host, error in offline_hosts:
            print(f"✗ {host}: {error}")

def print_availability_report(gpu_info_list):
    available_gpus = []
    
    print("\n=== AVAILABILITY REPORT ===")
    print("GPUs with < 100W power usage and < 100MiB memory usage:")
    
    for host_gpus in gpu_info_list:
        if not host_gpus:
            continue
            
        # Skip error entries
        if 'status' in host_gpus[0] and host_gpus[0]['status'] in ['error', 'timeout']:
            continue
            
        for info in host_gpus:
            if info['used_power'] < 100 and info['used_memory'] < 400:
                available_gpus.append(info)
                print(f"✓ {info['host']} - GPU {info['gpu']} ({info['name']})")
                print(f"  Memory: {info['used_memory']}MiB used / {info['total_memory']}MiB total")
                print(f"  Power: {info['used_power']}W used / {info['total_power']}W total")
    
    if not available_gpus:
        print("No GPUs available matching criteria.")
    
    return available_gpus

def look_for_gpu(host_flag):
    if host_flag == "0":
        hosts = computers = gpu_3090 + gpu_4070
    elif host_flag == "1":
        hosts = computers = gpu_3090
    elif host_flag == "2":
        hosts = computers = gpu_4070
        
    start_time = time.time()
    
    # Use ThreadPoolExecutor to run get_availability concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit tasks for each host and map them to their respective host
        future_to_host = {executor.submit(get_availability, host, 20): host for host in hosts}
        
        # Collect results as they complete
        results = []
        for future in concurrent.futures.as_completed(future_to_host):
            host = future_to_host[future]
            try:
                gpu_info = future.result()
                results.append(gpu_info)
            except Exception as exc:
                print(f"{host} generated an exception: {exc}")
                results.append([{'host': host, 'status': 'exception', 'error': str(exc)}])
    
    # Print all results
    print_gpu_info(results)
    
    # Print availability report
    available_gpus = print_availability_report(results)
    
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

    return [available_gpus[i]['host'] for i in range(len(available_gpus))]

def stop_all_processes(host_flag, timeout=20):
    """
    SSH into each host, check for Python processes belonging to the current user and kill them.
    
    Args:
        hosts (list): List of host names to connect to. Defaults to the predefined list.
        timeout (int): SSH connection timeout in seconds. Defaults to 20.
    
    Returns:
        list: Results of the operation on each host
    """
    if host_flag == "0":
        hosts = computers = gpu_3090 + gpu_4070
    elif host_flag == "1":
        hosts = computers = gpu_3090
    elif host_flag == "2":
        hosts = computers = gpu_4070


    if hosts is None:
        hosts = computers

    print(f"Stopping Python processes on {len(hosts)} hosts...")
    results = []
    
    def stop_processes_on_host(host):
        try:
            print(f"Connecting to {host}...")
            # First, list the processes
            result = subprocess.run(
                ['ssh', 
                 '-o', 'ServerAliveInterval=5',
                 '-o', 'ServerAliveCountMax=1',
                 host, 'ps aux | grep python | grep $USER'
                ],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            # Store the process info
            processes = result.stdout.strip().split('\n')
            processes = [p for p in processes if p and 'grep python' not in p]
            
            if not processes or (len(processes) == 1 and not processes[0]):
                return {
                    'host': host,
                    'status': 'success',
                    'message': 'No Python processes found',
                    'processes_killed': 0
                }
            
            # Kill the processes
            kill_result = subprocess.run(
                ['ssh',
                 '-o', 'ServerAliveInterval=5',
                 '-o', 'ServerAliveCountMax=1',
                 host, 'pkill -u $USER python'
                ],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return {
                'host': host,
                'status': 'success',
                'message': 'Killed Python processes',
                'processes_killed': len(processes),
                'process_list': processes
            }
            
        except subprocess.TimeoutExpired:
            print(f"Timeout connecting to {host} (exceeded {timeout} seconds)")
            return {
                'host': host, 
                'status': 'timeout', 
                'error': f'Connection timed out after {timeout}s'
            }
        
        except subprocess.CalledProcessError as e:
            # A return code of 1 from pkill means no processes were found (which is ok)
            if e.returncode == 1 and 'pkill' in str(e.cmd):
                return {
                    'host': host,
                    'status': 'success',
                    'message': 'No Python processes found to kill',
                    'processes_killed': 0
                }
            
            print(f"Error connecting to {host}: {e}")
            return {
                'host': host, 
                'status': 'error', 
                'error': str(e)
            }
        
        except Exception as e:
            print(f"Unexpected error with {host}: {e}")
            return {
                'host': host, 
                'status': 'error', 
                'error': str(e)
            }
    
    # Use ThreadPoolExecutor to run in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_host = {executor.submit(stop_processes_on_host, host): host for host in hosts}
        
        for future in concurrent.futures.as_completed(future_to_host):
            host = future_to_host[future]
            try:
                result = future.result()
                results.append(result)
                
                # Print results
                if result['status'] == 'success':
                    if result.get('processes_killed', 0) > 0:
                        print(f"✓ {host}: Killed {result['processes_killed']} Python processes")
                    else:
                        print(f"✓ {host}: No Python processes found")
                else:
                    print(f"✗ {host}: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"{host} generated an exception: {e}")
                results.append({
                    'host': host, 
                    'status': 'exception', 
                    'error': str(e)
                })
    
    # Summary
    killed_count = sum(r.get('processes_killed', 0) for r in results if r['status'] == 'success')
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = len(results) - success_count
    
    print(f"\n=== SUMMARY ===")
    print(f"Successfully connected to {success_count} hosts")
    print(f"Killed a total of {killed_count} Python processes")
    if error_count > 0:
        print(f"Failed to connect to {error_count} hosts")
    
    return results

if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser(description="Check GPU availability and stop processes on remote hosts.")
    parser.add_argument('--kill', type=bool, default=False, help="Stop all Python processes on remote hosts.")
    parser.add_argument('--hosts', type=str, default="0", help="List of hosts to check. If not provided, defaults to predefined list.", choices=["0", "1", "2"])
    # if len(sys.argv) > 1:
    args = parser.parse_args()    
    if args.kill == True:
        stop_all_processes(args.hosts)
        
    available_hosts = look_for_gpu(args.hosts)
