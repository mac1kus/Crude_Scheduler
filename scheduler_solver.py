class SolverPlanManager:
    """Manages solver-based tank filling plans"""
    
    def __init__(self, scheduler):
        self.scheduler = scheduler
        # Store reference to config for accessing params
        self.initial_params = getattr(scheduler, 'cfg', {})
    
    def initialize_solver_plan(self, params):
        """
        Initialize the solver's tank filling plan if available
        
        Args:
            params (dict): Configuration dictionary with solver_results
            
        Returns:
            bool: True if solver plan was successfully initialized, False otherwise
        """
        
        # STEP 1: Check if optimization is enabled
        use_optimized = params.get('use_optimized_schedule', False)
        pass
        
        if not use_optimized:
            pass
            return False
        
        # STEP 2: Try to get solver results from multiple sources
        solver_results = None
        
        # Source 1: Direct from params
        if 'solver_results' in params and params['solver_results']:
            solver_results = params['solver_results']
        
        # Source 2: From scheduler config
        elif hasattr(self.scheduler, 'solver_results') and self.scheduler.solver_results:
            solver_results = self.scheduler.solver_results
        
        # Source 3: From scheduler config dict
        elif hasattr(self.scheduler, 'cfg') and 'solver_results' in self.scheduler.cfg:
            solver_results = self.scheduler.cfg['solver_results']
            pass
        
        else:
            pass
            return False
        
        # STEP 3: Validate solver_results structure
        if not isinstance(solver_results, dict):
            pass
            return False
        
        
        # STEP 4: Extract tank filling plan
        tank_filling_plan = solver_results.get('tank_filling_plan', [])
        
        if not tank_filling_plan:
            pass
            return False
        
        
        # STEP 5: Store the tank filling plan
        self.scheduler.solver_tank_filling_plan = tank_filling_plan
        
        # STEP 6: Build cargo-to-tank assignment map
        self.scheduler.cargo_to_tank_assignments = {}
        self.scheduler.tank_assignment_progress = {}
        
        assignment_count = 0
        for assignment in tank_filling_plan:
            cargo_id = assignment.get('cargo_id')
            tank_id = assignment.get('tank_id')
            volume = assignment.get('volume', 0)
            crude_type = assignment.get('crude_type', 'Unknown')
            
            # Validate assignment structure
            if cargo_id is None or tank_id is None:
                continue
            
            # Initialize cargo assignments list if needed
            if cargo_id not in self.scheduler.cargo_to_tank_assignments:
                self.scheduler.cargo_to_tank_assignments[cargo_id] = []
            
            # Add assignment to cargo's list
            self.scheduler.cargo_to_tank_assignments[cargo_id].append({
                'tank_id': tank_id,
                'volume': volume,
                'crude_type': crude_type,
                'filled': 0  # Track how much has been filled
            })
            
            # Track progress per tank
            if tank_id not in self.scheduler.tank_assignment_progress:
                self.scheduler.tank_assignment_progress[tank_id] = {
                    'planned_fills': [],
                    'completed_fills': [],
                    'current_fill': None
                }
            
            self.scheduler.tank_assignment_progress[tank_id]['planned_fills'].append({
                'cargo_id': cargo_id,
                'crude_type': crude_type,
                'volume': volume
            })
            
            assignment_count += 1
        
        
        # STEP 7: Display summary
        
        # Show first few cargo assignments as sample
        pass
        for idx, (cargo_id, assignments) in enumerate(list(self.scheduler.cargo_to_tank_assignments.items())[:3]):
            pass
            for assignment in assignments[:2]:  # Show first 2 tanks per cargo
                pass
            if len(assignments) > 2:
                pass
        
        return True

    def get_all_solver_assignments_for_cargo(self, cargo_id):
        """
        Get ALL tank assignments for a cargo from solver plan
        
        Args:
            cargo_id (int): The cargo ID to look up
            
        Returns:
            list: List of assignment dicts, or empty list if not found
        """
        if not self.scheduler.cargo_to_tank_assignments:
            return []

        # Direct lookup
        if cargo_id in self.scheduler.cargo_to_tank_assignments:
            return self.scheduler.cargo_to_tank_assignments[cargo_id]
        
        # Fallback: try 1-based indexing (for legacy compatibility)
        cargo_keys = sorted(self.scheduler.cargo_to_tank_assignments.keys())
        if 1 <= cargo_id <= len(cargo_keys):
            actual_key = cargo_keys[cargo_id - 1]
            return self.scheduler.cargo_to_tank_assignments[actual_key]
        
        return []

    def process_cargo_filling_with_solver_plan(self, active_cargos, waiting_vessels, tanks, current_date, actual_date,
                                         pumping_rate_per_hour, tank_capacity, crude_mix_target,
                                         tanks_feeding_today, crude_mix_tolerance):
        """
        Process cargo filling following the solver's exact tank filling plan
        This replaces process_cargo_filling_fixed when solver plan is available
        FIXED: Continuous pumping across days and immediate tank refilling
        """
        from datetime import datetime, timedelta
        
        total_cargo_opening_stock = 0
        total_cargo_consumption_today = 0
        total_cargo_closing_stock = 0
        cargos_to_remove = []
        day_current_pumping_time = datetime.combine(current_date, datetime.min.time())

        for cargo_idx, active_cargo in enumerate(list(active_cargos)):
            cargo_opening_stock = active_cargo.get('remaining_volume', 0)
            active_cargo['last_pump_time'] = None
            cargo_consumption_today = 0

            if active_cargo['remaining_volume'] <= 0:
                continue

            # Skip pumping if cargo hasn't reached its pumping start time
            if current_date < active_cargo['pumping_start_time'].date():
                total_cargo_opening_stock += cargo_opening_stock
                total_cargo_closing_stock += active_cargo.get('remaining_volume', 0)
                continue

            cargo_id = active_cargo['cargo_id']
            cargo_crude_type = active_cargo.get('crude_type', 'Unknown')

            # ========== CRITICAL FIX #1: CONTINUOUS PUMPING ACROSS DAYS ==========
            current_pumping_time = active_cargo.get('continuous_pump_time', active_cargo['pumping_start_time'])
            start_of_today = datetime.combine(current_date, datetime.min.time())
            yesterday_end = start_of_today  # 00:00 today = end of yesterday (24:00 yesterday)

            # CRITICAL FIX: Don't reset to 00:00 if cargo was actively pumping yesterday
            # Check if continuous_pump_time is from yesterday and cargo still has volume
            if current_pumping_time.date() < current_date and active_cargo['remaining_volume'] > 0:
                # Cargo was pumping yesterday - continue from where it left off
                # Don't reset to start_of_today
                pass
            else:
                # Normal case: cargo starts fresh today
                current_pumping_time = max(current_pumping_time, start_of_today)

            # Calculate available pumping time for today
            day_end_time = datetime.combine(current_date, datetime.min.time()) + timedelta(days=1)

            # If pumping started yesterday, calculate from current_pumping_time to end of today
            if current_pumping_time < start_of_today:
                # Started yesterday, crosses midnight
                hours_to_pump_today = (day_end_time - current_pumping_time).total_seconds() / 3600
                pumping_start_this_day = current_pumping_time
            else:
                # Started today
                pumping_start_this_day = max(start_of_today, active_cargo['pumping_start_time'])
                hours_to_pump_today = (day_end_time - pumping_start_this_day).total_seconds() / 3600

            volume_to_pump_today = min(
                hours_to_pump_today * pumping_rate_per_hour,
                active_cargo['remaining_volume']
            )

            # ========== END OF FIX #1 ==========

            # Get ALL assignments for this cargo
            all_assignments = self.get_all_solver_assignments_for_cargo(cargo_id)

            if not all_assignments:
                # Fall back to finding any empty tank when solver plan exhausted
                empty_tank = self.scheduler.tank_manager._find_earliest_empty_tank(tanks, tanks_feeding_today)
                if empty_tank:
                    all_assignments = [{
                        'tank_id': empty_tank['id'],
                        'volume': active_cargo['remaining_volume'],
                        'crude_type': cargo_crude_type,
                        'filled': 0
                    }]
                    fallback_tank_name = self.scheduler.tank_manager.get_tank_display_name(empty_tank)
                    self.scheduler.alerts.append({
                        'type': 'info',
                        'day': actual_date.strftime('%d/%m'),
                        'message': f"Solver plan exhausted for Cargo {cargo_id}, using fallback {fallback_tank_name}"
                    })
                else:
                    self.scheduler.alerts.append({
                        'type': 'warning',
                        'day': actual_date.strftime('%d/%m'),
                        'message': f"No solver assignments found for Cargo {cargo_id} and no empty tanks available"
                    })
                    continue

            # Process each assignment for this cargo
            for assignment in all_assignments:

                tank_id = assignment['tank_id']
                # FIX: Check if this is a virtual tank trying to use a physical tank that's still busy
                target_tank = next((t for t in tanks if t['id'] == tank_id), None)
                if target_tank and target_tank.get('is_virtual'):
                    base_tank_id = target_tank.get('base_tank')

                    # Check if ANY tank (physical or virtual) using this base is still active
                    for other_tank in tanks:
                        # Check physical tank
                        if other_tank['id'] == base_tank_id and not other_tank.get('is_virtual'):
                            if other_tank['status'] != 'EMPTY':
                                continue  # Physical tank not empty, skip this assignment
                        # Check other virtual tanks of same base
                        if (other_tank.get('base_tank') == base_tank_id and
                            other_tank['id'] != tank_id and
                            other_tank['status'] in ['FEEDING', 'FILLING', 'SUSPENDED', 'SETTLING', 'LAB_TESTING', 'READY']):
                            continue  # Another virtual tank using this base is active

                planned_volume = assignment['volume']
                already_filled = assignment.get('filled', 0)
                remaining_to_fill = planned_volume - already_filled

                if remaining_to_fill <= 100:  # Already complete
                    continue

                # Get the target tank
                target_tank = next((t for t in tanks if t['id'] == tank_id), None)

                if not target_tank:
                    continue

                # ========== CRITICAL FIX #2: ONLY WAIT IF TANK IS ACTIVELY FEEDING RIGHT NOW ==========
                # Don't wait for tanks that emptied in the past - only wait if they're currently feeding
                # and will become available during this pumping session
                
                if target_tank['status'] == 'FEEDING' and target_tank.get('emptied_time_today'):
                    tank_available_time = target_tank['emptied_time_today']
                    
                    # Only wait if the tank will empty in the FUTURE (during this pumping session)
                    # If it already emptied in the past, don't wait - fill it now
                    if current_pumping_time < tank_available_time and tank_available_time.date() >= current_date:
                        # Tank is currently feeding and will empty soon - wait for it
                        idle_start = current_pumping_time
                        current_pumping_time = tank_available_time
                        self.scheduler.alerts.append({
                            'type': 'info', 
                            'day': actual_date.strftime('%d/%m'),
                            'message': f"Pumping from {active_cargo['vessel_name']} paused from {idle_start.strftime('%H:%M')} to {current_pumping_time.strftime('%H:%M')} waiting for {self.scheduler.tank_manager.get_tank_display_name(target_tank)} to finish feeding."
                        })
                # If tank is EMPTY, no waiting - fill it immediately at current_pumping_time
                # ========== END OF FIX #2 ==========

                # Check if this virtual tank's physical base is available
                if target_tank and target_tank.get('is_virtual'):
                    base_tank_id = target_tank.get('base_tank')

                    # Check if ANY other tank using this physical base is active
                    base_tank_busy = False
                    for other_tank in tanks:
                        if ((other_tank['id'] == base_tank_id and not other_tank.get('is_virtual')) or
                            (other_tank.get('base_tank') == base_tank_id and other_tank['id'] != tank_id)):
                            if other_tank['status'] in ['FEEDING', 'FILLING', 'SUSPENDED', 'SETTLING', 'LAB_TESTING', 'READY']:
                                base_tank_busy = True
                                break

                    if base_tank_busy:
                        continue  # Skip this assignment - physical tank is busy

                unavailable_statuses = ['FEEDING', 'SETTLING', 'LAB_TESTING', 'READY']
                if target_tank['status'] in unavailable_statuses:
                    continue  # This tank is busy with a later process, skip.

                # CRITICAL FIX: Check if another cargo is currently filling this tank
                if target_tank.get('currently_filling_by_cargo') and target_tank['currently_filling_by_cargo'] != cargo_id:
                    continue  # Another cargo is using this tank right now, skip to next assignment

                if target_tank['id'] in tanks_feeding_today or target_tank['fed_today']:
                    continue  # This tank was used for feeding today, skip.

                # Calculate current total system inventory
                current_system_total = sum(t['volume'] - t['dead_bottom'] for t in tanks if not t.get('is_virtual', False))
                system_wide_space = self.scheduler.total_system_usable_capacity - current_system_total

                allowed_volume, should_stop = self.scheduler.tank_manager.check_and_enforce_system_capacity(
                    tanks,
                    min(remaining_to_fill, volume_to_pump_today, active_cargo['remaining_volume']),
                    actual_date
                )
                if should_stop:
                    # System at capacity - stop ALL cargo filling immediately
                    return {
                        'cargo_opening_stock': total_cargo_opening_stock,
                        'cargo_consumption_today': total_cargo_consumption_today,
                        'cargo_closing_stock': total_cargo_closing_stock
                    }

                space_available = max(0, tank_capacity - target_tank['volume'])
                volume_to_fill = min(allowed_volume, space_available)

                if volume_to_fill <= 1:
                    continue

                # Log the discharge
                self.scheduler.daily_discharge_log.append({
                    'date': actual_date.strftime('%d/%m/%y'),
                    'cargo_type': active_cargo['vessel_name'],
                    'crude_type': active_cargo.get('crude_type', 'N/A'),
                    'tank_id': target_tank['id'], 
                    'volume_filled': volume_to_fill
                })

                # Record end of suspension
                if target_tank['status'] == 'SUSPENDED':
                    target_tank['suspended_end_datetime'] = current_pumping_time

                # Initialize tank for filling if needed
                if target_tank['status'] not in ['FILLING', 'SUSPENDED']:
                    target_tank['status'] = 'FILLING'
                    target_tank['filling_start_datetime'] = current_pumping_time

                    self.scheduler.filling_events_log.append({
                        'tank_id': target_tank['id'],
                        'start': current_pumping_time,
                        'end': None,
                        'settle_start': None,
                        'lab_start': None,
                        'ready_time': None,
                        'cargo_type': active_cargo['vessel_name'],
                        'start_level': target_tank['volume'],
                        'filled_volume': 0
                    })

                target_tank['currently_filling_by_cargo'] = cargo_id

                # Track crude composition
                if 'crude_composition' not in target_tank:
                    target_tank['crude_composition'] = {}
                if cargo_crude_type not in target_tank['crude_composition']:
                    target_tank['crude_composition'][cargo_crude_type] = 0

                # Perform the fill
                target_tank['crude_composition'][cargo_crude_type] += volume_to_fill
                target_tank['volume'] += volume_to_fill

                target_tank['available'] = max(0, target_tank['volume'] - target_tank['dead_bottom'])

                active_cargo['remaining_volume'] -= volume_to_fill
                volume_to_pump_today -= volume_to_fill
                cargo_consumption_today += volume_to_fill

                # Update assignment progress
                assignment['filled'] = already_filled + volume_to_fill

                # CRITICAL FIX: If this assignment is now complete, release the tank lock
                # so another cargo can continue filling this tank
                if assignment['filled'] >= planned_volume - 100:
                    pass
                    if target_tank.get('currently_filling_by_cargo') == cargo_id:
                        target_tank['currently_filling_by_cargo'] = None

                # Update tank mix percentages
                total_in_tank = max(1, target_tank['volume'] - target_tank['dead_bottom'])
                if total_in_tank > 0:
                    mix_percentages = {
                        crude: (vol / total_in_tank) * 100
                        for crude, vol in target_tank['crude_composition'].items()
                    }
                    target_tank['current_mix_percentages'] = mix_percentages

                # Capture start/end times
                start_time_for_log = current_pumping_time

                pumping_hours_spent = volume_to_fill / pumping_rate_per_hour if pumping_rate_per_hour > 0 else 0
                current_pumping_time += timedelta(hours=pumping_hours_spent)
                day_current_pumping_time = current_pumping_time

                end_time_for_log = current_pumping_time
                active_cargo['last_pump_time'] = end_time_for_log

                start_str = start_time_for_log.strftime('%H:%M')
                end_str = end_time_for_log.strftime('%H:%M')
                mix_str = ', '.join([f'{c}:{p:.1f}%' for c,p in target_tank.get('current_mix_percentages', {}).items()])

                tank_name = self.scheduler.tank_manager.get_tank_display_name(target_tank)
                self.scheduler.alerts.append({
                    'type': 'info',
                    'day': actual_date.strftime('%d/%m'),
                    'message': f"{tank_name}: Started filling at {start_str} hrs up to {end_str} hrs, {volume_to_fill:,.0f} bbl {cargo_crude_type} from {active_cargo['vessel_name']}. Mix now: {mix_str}"
                })

                # Check if tank is complete (usable capacity reached)
                usable_capacity = tank_capacity - target_tank['dead_bottom']
                if target_tank['volume'] >= usable_capacity - 1000:
                    # Check if already processed
                    if target_tank.get('completion_processed'):
                        continue
                    target_tank['completion_processed'] = True

                    settling_time_days = float(self.scheduler.initial_params.get('settlingTime', 2))
                    settling_time_hours = settling_time_days * 24

                    target_tank['status'] = 'SETTLING'
                    target_tank['settling_start_datetime'] = current_pumping_time
                    target_tank['settling_end_datetime'] = current_pumping_time + timedelta(hours=settling_time_hours)
                    target_tank['blend_complete'] = True
                    target_tank['currently_filling_by_cargo'] = None

                    tank_name = self.scheduler.tank_manager.get_tank_display_name(target_tank)
                    self.scheduler.alerts.append({
                        'type': 'info',
                        'day': actual_date.strftime('%d/%m'),
                        'message': f"{tank_name} FILLED at {current_pumping_time.strftime('%H:%M')}, starts SETTLING for {settling_time_days*24:.0f} hours until {target_tank['settling_end_datetime'].strftime('%d/%m %H:%M')}"
                    })

                    usable_volume = target_tank['volume'] - target_tank['dead_bottom']
                    self.scheduler.alerts.append({
                        'type': 'success',
                        'day': actual_date.strftime('%d/%m'),
                        'message': f"{tank_name} complete! Filled volume: {usable_volume:,.0f} bbl. Final mix: {', '.join([f'{c}:{p:.1f}%' for c,p in target_tank['current_mix_percentages'].items()])}"
                    })

                    # Update filling event log
                    settling_time_days = float(self.scheduler.initial_params.get('settlingTime', 2))
                    lab_testing_days = float(self.scheduler.initial_params.get('labTestingDays', 1))

                    for event in reversed(self.scheduler.filling_events_log):
                        if event['tank_id'] == target_tank['id'] and event['end'] is None:
                            event['end'] = current_pumping_time
                            event['settle_start'] = current_pumping_time
                            event['lab_start'] = current_pumping_time + timedelta(days=settling_time_days)
                            event['ready_time'] = current_pumping_time + timedelta(days=settling_time_days + lab_testing_days)
                            event['filled_volume'] = target_tank['volume']
                            break

            # Update totals
            total_cargo_opening_stock += cargo_opening_stock
            active_cargo['continuous_pump_time'] = current_pumping_time

            total_cargo_consumption_today += cargo_consumption_today
            total_cargo_closing_stock += active_cargo.get('remaining_volume', 0)

            # Remove cargo if empty
            if active_cargo['remaining_volume'] <= 1:
                cargos_to_remove.append(active_cargo)

                # Set suspended status on partially filled tanks
                last_tank_filled = next((tank for tank in tanks if tank.get('currently_filling_by_cargo') == cargo_id), None)
                if last_tank_filled and last_tank_filled['status'] == 'FILLING':
                    suspension_time = active_cargo.get('last_pump_time', current_pumping_time)

                    final_suspension_time = suspension_time if suspension_time else actual_date
                    last_tank_filled['status'] = 'SUSPENDED'
                    last_tank_filled['suspended_start_datetime'] = final_suspension_time
                    last_tank_filled['currently_filling_by_cargo'] = None

                    tank_name = self.scheduler.tank_manager.get_tank_display_name(last_tank_filled)
                    self.scheduler.alerts.append({
                        'type': 'warning', 
                        'day': actual_date.strftime('%d/%m'),
                        'message': f"{tank_name} SUSPENDED at {final_suspension_time.strftime('%H:%M')}. Awaiting more crude."
                    })

        # Remove empty cargos and free berths
        last_discharged_cargo = None
        berth_discharge_times = {}  
        for cargo in cargos_to_remove:
            if cargo in active_cargos:
                active_cargos.remove(cargo)
                last_discharged_cargo = cargo
                berth_id = cargo.get('berth_id')
                if berth_id and self.scheduler.berth_status[berth_id]['cargo_id'] == cargo['cargo_id']:
                    discharge_time = cargo.get('last_pump_time', datetime.combine(actual_date.date(), datetime.min.time()))
                    discharge_time_str = discharge_time.strftime('%H:%M') if discharge_time else "00:00"

                    self.scheduler.berth_status[berth_id] = {'occupied': False, 'vessel': None, 'cargo_id': None}
                    self.scheduler.berth_last_freed[berth_id] = discharge_time
                    berth_discharge_times[berth_id] = discharge_time
                
                    self.scheduler.alerts.append({
                        'type': 'info',
                        'day': actual_date.strftime('%d/%m'),
                        'message': f"{cargo['vessel_name']} finished discharge at {discharge_time_str}. Berth {berth_id} is now free."
                    })

        # Check for waiting vessels
        for berth_id in self.scheduler.berth_status:
            if not self.scheduler.berth_status[berth_id]['occupied'] and waiting_vessels:
                next_vessel = waiting_vessels.pop(0)
                self.scheduler.berth_status[berth_id]['occupied'] = True
                self.scheduler.berth_status[berth_id]['vessel'] = next_vessel['vessel_name']
                self.scheduler.berth_status[berth_id]['cargo_id'] = next_vessel['cargo_id']

                new_cargo = next_vessel.copy()
                new_cargo['berth_id'] = berth_id
                new_cargo['remaining_volume'] = new_cargo['size']

                berth_clearance_hours = 0
                if berth_id in berth_discharge_times:
                    new_arrival_time = berth_discharge_times[berth_id] + timedelta(hours=berth_clearance_hours)
                else:
                    new_arrival_time = actual_date + timedelta(hours=berth_clearance_hours)

                new_cargo['arrival_datetime'] = new_arrival_time
                new_cargo['pumping_start_time'] = new_arrival_time + timedelta(days=float(self.scheduler.initial_params.get('preDischargeDays', 1)))

                active_cargos.append(new_cargo)

        return {
            'cargo_opening_stock': total_cargo_opening_stock,
            'cargo_consumption_today': total_cargo_consumption_today,
            'cargo_closing_stock': total_cargo_closing_stock
        }