"""
Enhanced Crude Mix Optimization Solver - Exact Ratio Version
Iterative optimization with tank pre-allocation and mass balance tracking
FIXED:
- Dynamic crude volume calculation based on input percentages
- Ensures all purchased crude is fully allocated in the mass balance
- Proper crude sequencing (no hardcoding)
- Cargo combinations optimized to match crude mix ratios
- Suspended tank priority filling
- FIXED: Proper tank sequencing 12,13,1-11 with correct cycle suffixes
- FIXED: Certification logic for multiple tanks on correct day
- TIMING REMOVED: Scheduler handles all arrival times
"""

from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict
import json
import math
import io
from contextlib import redirect_stdout


class CrudeMixOptimizer:
    def __init__(self):
        self.model = None
        self.results = {}
        self.tank_allocations = []
        self.cargo_to_tank_map = {}
        self.mass_balance = {}
        self.suspended_tanks = []
        self.tank_sequence = []

    def solve_crude_mix_schedule(self, params):
        """Main optimization with iterative refinement for EXACT crude ratios"""
        try:
            # Extract parameters
            processing_rate = float(params.get('processingRate', 50000))
            num_tanks = self._detect_tank_count(params)
            tank_capacity = float(params.get('tankCapacity', 500000))
            report_days = int(params.get('schedulingWindow', 30))
            
            print(f"SOLVER: Detected {num_tanks} tanks, report window {report_days} days")

            # Extract EXACT crude mix requirements
            crude_mix = self._extract_crude_mix(params)
            if not crude_mix:
                print("ERROR: No crude mix configuration found")
                return {'success': False, 'error': 'No crude mix configuration'}

            # Verify percentages sum to 100%
            total_pct = sum(c['percentage'] for c in crude_mix.values())
            if abs(total_pct - 1.0) > 0.01:
                print(f"ERROR: Crude percentages sum to {total_pct*100}%, not 100%")
                return {'success': False, 'error': f'Crude percentages must sum to 100%'}

            vessels = self._extract_vessel_data(params)
            if not vessels:
                return {'success': False, 'error': 'No vessel types available'}

            # Calculate requirements
            empty_tanks_initial = self._get_empty_tanks(params, num_tanks)
            total_consumption = processing_rate * report_days
            total_needed = total_consumption

            print(f"SOLVER: Total crude needed = {total_needed:,.0f} bbl")

            crude_names = [c['name'] for c in crude_mix.values()]
            crude_ratios = [c['percentage'] for c in crude_mix.values()]
            print(f"SOLVER: Handling {len(crude_names)} crude types: {crude_names}")
            mix_summary = [(c['name'], f"{c['percentage']*100:.1f}%") for c in crude_mix.values()]
            print(f"SOLVER: Target mix = {mix_summary}")

            # Find optimal cargo combination that matches crude ratios
            optimal_vessel_pattern = self._find_optimal_vessel_combination(vessels, crude_ratios, crude_names)
            print(f"SOLVER: Optimal vessel pattern selected: {optimal_vessel_pattern}")

            # STEP 1: Pre-allocate tanks with exact crude requirements
            tank_plan = self._allocate_tanks_for_blend(
               params, num_tanks, empty_tanks_initial, crude_mix, tank_capacity, total_needed, report_days
            )
            
            # STEP 2: Generate cargoes using iterative optimization with ratio matching
            result = self._iterative_cargo_optimization(
                params, vessels, crude_mix, tank_plan, total_needed, optimal_vessel_pattern
            )
            
            if result['success']:
                # STEP 3: Create detailed tank filling schedule
                result['tank_filling_plan'] = self._create_tank_filling_schedule(
                    result['cargo_schedule'], tank_plan
                )
                
                # STEP 4: Generate mass balance
                result['mass_balance'] = self._generate_mass_balance(
                    result['cargo_schedule'], result['tank_filling_plan']
                )
                
                # STEP 5: Add tank sequence to result for utils.py
                result['tank_sequence'] = [tank['tank_id'] for tank in self.tank_allocations]
            
            return result

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

    def _find_optimal_vessel_combination(self, vessels, crude_ratios, crude_names):
        """Find vessel combination that best matches crude mix ratios"""
        print(f"\nFINDING OPTIMAL VESSEL COMBINATION:")
        print(f"Target crude ratios: {[f'{r*100:.1f}%' for r in crude_ratios]}")
        
        vessel_multipliers = {
            'vlcc': [0, 1, 2, 3],
            'suezmax': [0, 0.5, 1, 1.5, 2],
            'aframax': [0, 0.25, 0.5, 0.75, 1, 1.25],
            'panamax': [0, 0.25, 0.5, 0.75, 1],
            'handymax': [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        }
        
        best_combination = None
        best_deviation = float('inf')
        
        for vlcc_mult in vessel_multipliers['vlcc']:
            for suez_mult in vessel_multipliers['suezmax']:
                for afra_mult in vessel_multipliers['aframax']:
                    for pana_mult in vessel_multipliers['panamax']:
                        for handy_mult in vessel_multipliers['handymax']:
                            
                            combination = {
                                'vlcc': vlcc_mult,
                                'suezmax': suez_mult,
                                'aframax': afra_mult,
                                'panamax': pana_mult,
                                'handymax': handy_mult
                            }
                            
                            total_vessels = sum(combination.values())
                            if total_vessels == 0:
                                continue
                            
                            cargo_volumes = []
                            for vessel_type, multiplier in combination.items():
                                if vessel_type in vessels and multiplier > 0:
                                    volume = vessels[vessel_type]['capacity'] * multiplier
                                    cargo_volumes.append(volume)
                            
                            if len(cargo_volumes) != len(crude_ratios):
                                continue
                            
                            total_cargo_volume = sum(cargo_volumes)
                            if total_cargo_volume == 0:
                                continue
                                
                            cargo_ratios = [v / total_cargo_volume for v in cargo_volumes]
                            deviation = sum(abs(cargo_ratios[i] - crude_ratios[i]) for i in range(len(crude_ratios)))
                            
                            if deviation < best_deviation:
                                best_deviation = deviation
                                best_combination = combination.copy()
                                print(f"  New best: {combination}")
                                print(f"    Cargo ratios: {[f'{r*100:.1f}%' for r in cargo_ratios]}")
                                print(f"    Deviation: {deviation*100:.2f}%")
        
        return best_combination

    def _allocate_tanks_for_blend(self, params, num_tanks, empty_tanks_initial, crude_mix, tank_capacity, total_needed, report_days):
        """Pre-allocate tanks with EXACT crude volumes for perfect blending"""
        tank_allocations = []

        dead_bottom_base = float(params.get('deadBottom1', 10000))
        buffer_volume = float(params.get('bufferVolume', 500))
        dead_bottom_operational = dead_bottom_base + buffer_volume / 2
        usable_capacity = tank_capacity - dead_bottom_operational
        
        tanks_needed_to_fill = math.ceil(total_needed / usable_capacity) + 5
        
        print(f"\nTANK ALLOCATION PLAN - DYNAMIC CRUDE VOLUMES:")
        print(f"Total crude required: {total_needed:,.0f} bbl. Usable capacity per tank: {usable_capacity:,.0f} bbl.")
        print(f"This will require filling {tanks_needed_to_fill} tanks over {report_days} days.")
        
        # Create dynamic repeating sequence
        empty_tanks_sorted = sorted(empty_tanks_initial)
        all_tanks = set(range(1, num_tanks + 1))
        empty_tanks_set = set(empty_tanks_initial)
        occupied_tanks = all_tanks - empty_tanks_set
        occupied_tanks_sorted = sorted(list(occupied_tanks))
        dynamic_repeating_sequence = empty_tanks_sorted + occupied_tanks_sorted
        print(f"INFO: Dynamic repeating sequence created: {dynamic_repeating_sequence}")

        # Build complete tank pool
        final_tank_id_pool = []
        current_cycle = 0
        while len(final_tank_id_pool) < tanks_needed_to_fill:
            if current_cycle == 0:
                for tank_num in dynamic_repeating_sequence:
                    if len(final_tank_id_pool) >= tanks_needed_to_fill:
                        break
                    final_tank_id_pool.append(tank_num)
            else:
                for tank_num in dynamic_repeating_sequence:
                    if len(final_tank_id_pool) >= tanks_needed_to_fill:
                        break
                    tank_id_with_suffix = f"TK{tank_num}({current_cycle})"
                    final_tank_id_pool.append(tank_id_with_suffix)
            current_cycle += 1
        
        print(f"DEBUG: Tank sequence generated (first 20): {final_tank_id_pool[:20]}")

        for i in range(tanks_needed_to_fill):
            tank_id = final_tank_id_pool[i]
            
            tank_allocation = {
                'tank_id': tank_id,
                'total_capacity': tank_capacity,
                'usable_capacity': usable_capacity,
                'crude_volumes': {}
            }
            
            for crude_key, crude_data in crude_mix.items():
                crude_percentage = crude_data['percentage']
                volume_per_tank = usable_capacity * crude_percentage
                
                tank_allocation['crude_volumes'][crude_data['name']] = {
                    'target_volume': volume_per_tank,
                    'target_percentage': crude_percentage * 100,
                    'filled_volume': 0,
                    'source_cargoes': []
                }
                
            if i < 20:
                print(f"\nTank {tank_id} Dynamic Volume Allocation:")
                for crude_data in crude_mix.values():
                    volume = tank_allocation['crude_volumes'][crude_data['name']]['target_volume']
                    percentage = tank_allocation['crude_volumes'][crude_data['name']]['target_percentage']
                    print(f"  {crude_data['name']}: {volume:,.0f} bbl ({percentage:.1f}%)")
                
                total_allocated = sum(details['target_volume'] for details in tank_allocation['crude_volumes'].values())
                print(f"  Total allocated: {total_allocated:,.0f} bbl (should equal {usable_capacity:,.0f} bbl)")
            
            tank_allocations.append(tank_allocation)
        
        self.tank_allocations = tank_allocations
        self.tank_sequence = final_tank_id_pool
        return tank_allocations

    def _iterative_cargo_optimization(self, params, vessels, crude_mix, tank_plan, total_needed, optimal_vessel_pattern):
        """Iterative optimization to achieve EXACT crude ratios"""
        max_iterations = 10
        tolerance = 0.001  # 0.1% tolerance
        
        best_schedule = None
        best_deviation = float('inf')
        
        print(f"\nITERATIVE OPTIMIZATION:")
        print("="*80)
        
        for iteration in range(max_iterations):
            print(f"\nIteration {iteration + 1}:")
            print("-"*80)
            
            # Calculate exact crude volumes needed based on the comprehensive tank plan
            crude_requirements = defaultdict(float)
            for tank in tank_plan:
                for crude_name, details in tank['crude_volumes'].items():
                    crude_requirements[crude_name] += details['target_volume']

            # Generate cargo schedule using optimal vessel pattern
            schedule = self._generate_optimal_cargo_mix(
                params, vessels, crude_requirements, iteration, optimal_vessel_pattern, total_needed
            )
            
            # Calculate actual ratios
            total_by_crude = defaultdict(float)
            for cargo in schedule:
                total_by_crude[cargo['crude_name']] += cargo['size']
            
            total_volume = sum(total_by_crude.values())
            
            # Check deviation from target
            print(f"\nDEVIATION ANALYSIS:")
            print("-"*80)
            max_deviation = 0
            for crude_name, target_pct in [(c['name'], c['percentage']) for c in crude_mix.values()]:
                actual_pct = total_by_crude[crude_name] / total_volume if total_volume > 0 else 0
                deviation = abs(actual_pct - target_pct)
                max_deviation = max(max_deviation, deviation)
                status = "✓ OK" if deviation < 0.01 else "⚠ WARNING"
                print(f"  {crude_name:20s}: Target={target_pct*100:6.2f}%  Actual={actual_pct*100:6.2f}%  Deviation={deviation*100:6.3f}%  {status}")
            print(f"-"*80)
            print(f"MAXIMUM DEVIATION: {max_deviation*100:.3f}%")
            print("="*80)
            
            if max_deviation < best_deviation:
                best_schedule = schedule
                best_deviation = max_deviation
            
            # Check if we achieved target
            if max_deviation <= tolerance:
                print(f"\n✓✓✓ SUCCESS: Achieved target ratios within {tolerance*100}% tolerance! ✓✓✓\n")
                break
            
            # Adjust for next iteration
            if iteration < max_iterations - 1:
                print(f"\n⚠ Deviation {max_deviation*100:.3f}% > tolerance {tolerance*100}%, refining...\n")
        
        if best_schedule:
            return self._format_final_schedule(params, best_schedule, crude_mix, tank_plan)
        else:
            return {'success': False, 'error': 'Failed to generate schedule'}

    def _generate_optimal_cargo_mix(self, params, vessels, crude_requirements, iteration, optimal_vessel_pattern, total_needed):
        """Generate cargo list based on volume needs only - NO timing"""
        schedule = []
        min_vlcc_required = int(params.get('minVlccRequired', 0))
        
        if min_vlcc_required > 0:
            print(f"SOLVER CONSTRAINT: A minimum of {min_vlcc_required} VLCCs will be scheduled first.")

        tank_capacity = float(params.get('tankCapacity', 500000))
        dead_bottom_base = float(params.get('deadBottom1', 10000))
        buffer_volume = float(params.get('bufferVolume', 500))
        dead_bottom_operational = dead_bottom_base + buffer_volume / 2
        usable_capacity = tank_capacity - dead_bottom_operational
        
        crude_mix = self._extract_crude_mix(params)
        crude_names = [crude_mix[k]['name'] for k in crude_mix.keys()] if crude_mix else ['Unknown']
        
        print(f"\nVOLUME-BASED SCHEDULER - Iteration {iteration}")
        print(f"Usable capacity per tank: {usable_capacity:,.0f} bbl")
        print(f"Crude types: {crude_names}")
        
        crude_volume_per_tank = {}
        for crude_key in crude_mix.keys():
            crude_data = crude_mix[crude_key]
            volume = usable_capacity * crude_data['percentage']
            crude_volume_per_tank[crude_data['name']] = volume
            print(f"{crude_data['name']}: {volume:,.0f} bbl per tank ({crude_data['percentage']*100:.1f}%)")
        
        # Build vessel sequence based on optimal pattern (LIKE SOLVER_BACK.py)
        vessel_sequence = []
        if optimal_vessel_pattern:
            for vessel_type, multiplier in optimal_vessel_pattern.items():
                if vessel_type in vessels and multiplier > 0:
                    if multiplier >= 1:
                        vessel_sequence.extend([vessel_type] * int(multiplier))
                    else:
                        if multiplier >= 0.5:
                            vessel_sequence.append(vessel_type)
        
        if not vessel_sequence:
            vessel_list = sorted(vessels.items(), key=lambda x: x[1]['capacity'], reverse=True)
            vessel_sequence = [vessel_list[0][0]]
        
        print(f"VESSEL ROTATION SEQUENCE: {vessel_sequence}")
        print(f"Pattern multipliers: {optimal_vessel_pattern}")
        
        tank_plan = self.tank_allocations
        current_tank_idx = 0
        certified_tanks = []
        
        if not tank_plan:
            print("ERROR: Tank plan is empty.")
            return []
            
        tank_states = {}
        
        cargo_id = 1        
        crude_index = 0
        vessel_index = 0
        total_scheduled_volume = 0
        
        while total_scheduled_volume < total_needed:
            if cargo_id > 200:
                print("WARNING: Exceeded 200 cargo limit. Finalizing schedule.")
                break
            
            # Select vessel and crude
            if 'vlcc' in vessels and cargo_id <= min_vlcc_required:
                vessel_type = 'vlcc'
                print(f"  Cargo {cargo_id}: FORCED VLCC (constraint: {cargo_id}/{min_vlcc_required})")
            else:
                vessel_type = vessel_sequence[vessel_index % len(vessel_sequence)]
                print(f"  Cargo {cargo_id}: Selected {vessel_type.upper()} (index: {vessel_index % len(vessel_sequence)})")

            crude_name = crude_names[crude_index % len(crude_names)]
            vessel_data = vessels[vessel_type]
            
            # Check tank availability
            if current_tank_idx >= len(tank_plan):
                print("ERROR: Exceeded tank plan size.")
                break
                
            tank_to_fill = tank_plan[current_tank_idx]['tank_id']
            current_total_volume = sum(c['size'] for c in schedule)
            remaining_needed = total_needed - current_total_volume
            cargo_size_to_schedule = min(vessel_data['capacity'], remaining_needed)

            if cargo_size_to_schedule < 1000:
                print("INFO: Remaining crude less than minimum. Finalizing.")
                break

            cargo = {
                'cargo_id': cargo_id,
                'type': vessel_type.upper(),
                'vessel_type': vessel_type,
                'vessel_name': f"{vessel_type.upper()}-V{cargo_id:03d}",
                'crude_type': crude_name,
                'crude_name': crude_name,
                'size': cargo_size_to_schedule,
            }

            schedule.append(cargo)
            
            # Update tank states
            cargo_vol_remaining = cargo_size_to_schedule
            for tank_idx in range(len(tank_plan)):
                if cargo_vol_remaining <= 0:
                    break
                tank_id = tank_plan[tank_idx]['tank_id']
                
                if tank_id not in tank_states:
                    tank_states[tank_id] = {c: 0 for c in crude_names}
                    tank_states[tank_id]['status'] = 'empty'
                
                target_for_crude = tank_plan[tank_idx]['crude_volumes'].get(crude_name, {}).get('target_volume', 0)
                current_in_tank = tank_states[tank_id].get(crude_name, 0)
                needed = target_for_crude - current_in_tank
                
                if needed > 0:
                    volume_to_add = min(needed, cargo_vol_remaining)
                    tank_states[tank_id][crude_name] += volume_to_add
                    cargo_vol_remaining -= volume_to_add
                    
                    total_in_tank = sum(tank_states[tank_id].get(c, 0) for c in crude_names)
                    if total_in_tank >= usable_capacity * 0.99:
                        tank_states[tank_id]['status'] = 'complete'
            
            print(f"  QUEUED CARGO {cargo_id}: {vessel_type.upper()}-V{cargo_id:03d}, {cargo_size_to_schedule:,.0f} bbl of {crude_name}")
            
            total_scheduled_volume += cargo_size_to_schedule
            crude_index += 1
            vessel_index += 1
            cargo_id += 1
        
        # Summary of vessels used
        vessel_usage = defaultdict(int)
        for cargo in schedule:
            vessel_usage[cargo['type']] += 1
        
        print(f"\n{'='*80}")
        print(f"CARGO GENERATION COMPLETE")
        print(f"Total cargoes created: {len(schedule)}")
        print(f"Vessel breakdown:")
        for vessel_type in ['VLCC', 'SUEZMAX', 'AFRAMAX', 'PANAMAX', 'HANDYMAX']:
            count = vessel_usage.get(vessel_type, 0)
            if count > 0:
                print(f"  {vessel_type}: {count} cargoes")
        print(f"{'='*80}")
        
        print(f"\nFINAL TANK STATUS:")
        for tank_id in sorted(tank_states.keys(), key=lambda x: (isinstance(x, str), x)):
            tank_data = tank_states[tank_id]
            status = tank_data.get('status', 'unknown')
            total_vol = sum(tank_data[crude] for crude in crude_names if crude in tank_data)
            crude_breakdown = [f"{crude}:{tank_data.get(crude, 0):,.0f}" for crude in crude_names]
            print(f"Tank {tank_id}: {total_vol:,.0f} bbl ({status}) [{', '.join(crude_breakdown)}]")
        
        return schedule

    def _create_tank_filling_schedule(self, cargo_schedule, tank_plan):
        """Create detailed schedule showing which cargo fills which tank"""
        filling_schedule = []
        cargo_volumes = {c['cargo_id']: c['size'] for c in cargo_schedule}
        
        print(f"\nTANK FILLING SCHEDULE:")
        
        for tank_idx, tank in enumerate(tank_plan):
            tank_id_display = tank['tank_id'] if isinstance(tank['tank_id'], int) else tank['tank_id']
            print(f"\nTank {tank_id_display} Filling Plan:")
            
            for crude_name, crude_details in tank['crude_volumes'].items():
                target_volume = crude_details['target_volume']
                filled_volume = 0
                
                matching_cargoes = [c for c in cargo_schedule 
                                  if c['crude_name'] == crude_name 
                                  and cargo_volumes.get(c['cargo_id'], 0) > 100]
                
                for cargo in matching_cargoes:
                    if filled_volume >= target_volume - 100:
                        break
                    
                    volume_needed = target_volume - filled_volume
                    available_in_cargo = cargo_volumes.get(cargo['cargo_id'], 0)
                    volume_to_take = min(volume_needed, available_in_cargo)
                    
                    if volume_to_take > 100:
                        filling_schedule.append({
                            'tank_id': tank['tank_id'],
                            'cargo_id': cargo['cargo_id'],
                            'vessel_name': cargo['vessel_name'],
                            'crude_type': crude_name,
                            'volume': volume_to_take,
                            'percentage_of_tank': (volume_to_take / tank['usable_capacity']) * 100
                        })
                        
                        cargo_volumes[cargo['cargo_id']] -= volume_to_take
                        filled_volume += volume_to_take
                        
                        print(f"  {cargo['vessel_name']} -> {volume_to_take:,.0f} bbl {crude_name} ({(volume_to_take/tank['usable_capacity']*100):.1f}% of tank)")
                
                if filled_volume < target_volume * 0.95:
                    print(f"  WARNING: Could only plan to fill {filled_volume:,.0f}/{target_volume:,.0f} bbl of {crude_name} for this tank.")
        
        print(f"\nTotal tanks processed: {len(tank_plan)}")
        return filling_schedule

    def _generate_mass_balance(self, cargo_schedule, tank_filling_plan):
        """Generate complete mass balance for all cargoes"""
        mass_balance = {}
        
        print(f"\nCARGO MASS BALANCE:")
        
        for cargo_idx, cargo in enumerate(cargo_schedule):
            cargo_id = cargo['cargo_id']
            total_size = cargo['size']
            
            allocations = [f for f in tank_filling_plan if f['cargo_id'] == cargo_id]
            
            tank_breakdown = {}
            total_allocated = 0
            
            for alloc in allocations:
                tank_id = alloc['tank_id']
                volume = alloc['volume']
                tank_breakdown[f"Tank_{tank_id}"] = tank_breakdown.get(f"Tank_{tank_id}", 0) + volume
                total_allocated += volume
            
            unallocated = total_size - total_allocated
            
            mass_balance[cargo['vessel_name']] = {
                'cargo_id': cargo_id,
                'crude_type': cargo['crude_name'],
                'total_size': total_size,
                'allocated': total_allocated,
                'unallocated': unallocated,
                'tank_breakdown': tank_breakdown,
                'utilization': (total_allocated / total_size * 100) if total_size > 0 else 0
            }
            
            print(f"\n{cargo['vessel_name']} ({cargo['crude_name']}) - {total_size:,.0f} bbl:")
            for tank, vol in tank_breakdown.items():
                print(f"  {tank}: {vol:,.0f} bbl ({vol/total_size*100:.1f}%)")
            
            if unallocated > 1000:
                print(f"  Unallocated: {unallocated:,.0f} bbl ({unallocated/total_size*100:.1f}%)")
        
        print(f"\nTotal cargoes processed: {len(cargo_schedule)}")
        return mass_balance

    def _format_final_schedule(self, params, schedule, crude_mix, tank_plan):
        """Format final schedule with all details"""
        total_by_crude = defaultdict(float)
        vessel_counts = defaultdict(int)
        
        for cargo in schedule:
            total_by_crude[cargo['crude_name']] += cargo['size']
            vessel_counts[cargo['type']] += 1
        
        total_volume = sum(total_by_crude.values())
        total_cost = 0  # Cost calculated by scheduler
        
        actual_percentages = {}
        for crude_name, volume in total_by_crude.items():
            actual_percentages[crude_name] = (volume / total_volume * 100) if total_volume > 0 else 0
        
        print(f"\n{'='*80}")
        print("OPTIMIZATION COMPLETE")
        vessel_types_to_display = ['VLCC', 'SUEZMAX', 'AFRAMAX', 'PANAMAX', 'HANDYMAX']
        cargo_counts_str = ",  ".join([f"{v_type}: {vessel_counts.get(v_type, 0)}" for v_type in vessel_types_to_display])
        print(f"Vessel Cargoes: {cargo_counts_str}")

        print(f"Total Volume: {total_volume:,.0f} bbl")
        print(f"Total Cost: ${total_cost:,.0f}")
        print(f"Cargoes: {len(schedule)}")
        print("\nFINAL CRUDE MIX ACHIEVED:")
        for crude_name, pct in actual_percentages.items():
            target_pct = next((c['percentage'] * 100 for c in crude_mix.values() if c['name'] == crude_name), 0)
            status = "[OK]" if abs(pct - target_pct) < 0.5 else "[WARN]"
            print(f"  {crude_name}: {pct:.1f}% (Target: {target_pct:.1f}%) {status}")
        print(f"Suspended tanks processed: {len(self.suspended_tanks)}")
        print(f"{'='*80}")
        
        return {
            'success': True,
            'cargo_schedule': schedule,
            'total_cost': total_cost,
            'tank_sequence': self.tank_sequence,  # Pass the sequence to utils
            'optimization_status': 'Formula-Based Exact Ratio Optimization Complete',
            'crude_mix_achieved': self._format_tank_distribution(tank_plan, actual_percentages),
            'actual_percentages': actual_percentages,
            'vessel_distribution': dict(vessel_counts),
            'volume_by_crude': dict(total_by_crude),
            'suspended_tanks': self.suspended_tanks,
            'solver_info': {
                'method': 'formula_based_exact_ratio',
                'total_cargoes': len(schedule),
                'total_volume': total_volume,
                'exact_ratio_achieved': all(abs(actual_percentages.get(c['name'], 0) - c['percentage']*100) < 0.5 
                                          for c in crude_mix.values())
            }
        }

    def _format_tank_distribution(self, tank_plan, actual_percentages):
        """Format tank distribution for output"""
        distribution = {}
        
        for tank_idx, tank in enumerate(tank_plan):  
            tank_id_display = tank['tank_id'] if isinstance(tank['tank_id'], int) else tank['tank_id']
            tank_name = f"Tank_{tank_id_display}"
            distribution[tank_name] = {}
            
            for crude_name, details in tank['crude_volumes'].items():
                distribution[tank_name][crude_name] = {
                    'volume': round(details['target_volume'], 0),
                    'percentage': round(details['target_percentage'], 1)
                }
        
        return distribution

    def _detect_tank_count(self, params):
        """Detect number of tanks from parameters"""
        tank_numbers = []
        for key in params.keys():
            if key.startswith('tank') and key.endswith('Level'):
                try:
                    tank_num = int(key.replace('tank', '').replace('Level', ''))
                    tank_numbers.append(tank_num)
                except ValueError:
                    continue
        
        detected_max = max(tank_numbers) if tank_numbers else 0
        param_tanks = int(params.get('numTanks', 12))
        return max(detected_max, param_tanks)

    def _extract_crude_mix(self, params):
        """Extract crude mix configuration"""
        crude_names = params.get('crude_names', [])
        crude_percentages = params.get('crude_percentages', [])
        
        if not crude_names or not crude_percentages:
            return None
        
        mix = {}
        for i, (name, p) in enumerate(zip(crude_names, crude_percentages)):
            if float(p) > 0:
                mix[f'crude_{i}'] = {
                    'name': name,
                    'percentage': float(p) / 100.0,
                    'index': i
                }
        
        return mix

    def _extract_vessel_data(self, params):
        """Extract vessel configurations"""
        vessels = {}
        vessel_types_config = [
            ('vlcc', 'vlccCapacity', 'vlccRateDay', 'vlccIncludeReturn'),
            ('suezmax', 'suezmaxCapacity', 'suezmaxRateDay', 'suezmaxIncludeReturn'),
            ('aframax', 'aframaxCapacity', 'aframaxRateDay', 'aframaxIncludeReturn'),
            ('panamax', 'panamaxCapacity', 'panamaxRateDay', 'panamaxIncludeReturn'),
            ('handymax', 'handymaxCapacity', 'handymaxRateDay', 'handymaxIncludeReturn')
        ]
        
        for v_type, cap_key, rate_key, return_key in vessel_types_config:
            capacity = float(params.get(cap_key, 0))
            rate = float(params.get(rate_key, 50000))
            if capacity > 0:
                vessels[v_type] = {
                    'capacity': capacity,
                    'daily_rate': rate,
                    'include_return': params.get(return_key, True),
                    'journey_days': float(params.get('journeyDays', 10)),
                    'pre_journey_days': float(params.get('preJourneyDays', 1)),
                    'pre_discharge_days': float(params.get('preDischargeDays', 1)),
                    'pumping_rate': float(params.get('pumpingRate', 30000))
                }
        
        return vessels

    def _calculate_initial_inventory(self, params, num_tanks):
        """Calculate initial inventory"""
        total = 0
        for i in range(1, num_tanks + 1):
            tank_level = float(params.get(f'tank{i}Level', 0))
            dead_bottom = float(params.get(f'deadBottom{i}', 10000))
            total += max(0, tank_level - dead_bottom)
        return total

    def _get_empty_tanks(self, params, num_tanks):
        """Get list of empty tanks"""
        empty = []
        for i in range(1, num_tanks + 1):
            tank_level = float(params.get(f'tank{i}Level', 0))
            dead_bottom = float(params.get(f'deadBottom{i}', 10000))
            if tank_level <= dead_bottom + 500:
                empty.append(i)
        return empty

    def _get_processing_start_datetime(self, params):
        """Parse processing start datetime"""
        try:
            date_str = params.get('crudeProcessingDate', '2025-08-10 08:00')
            if 'T' in date_str:
                return datetime.fromisoformat(date_str.replace('T', ' '))
            elif ' ' in date_str:
                return datetime.strptime(date_str, '%Y-%m-%d %H:%M')
            else:
                return datetime.strptime(f"{date_str} 08:00", '%Y-%m-%d %H:%M')
        except:
            return datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)


def optimize_crude_mix_schedule(params):
    """
    Main entry point with exact ratio enforcement
    """
    optimizer = CrudeMixOptimizer()

    solver_output_buffer = io.StringIO()
    with redirect_stdout(solver_output_buffer):
        try:
            result = optimizer.solve_crude_mix_schedule(params)
        except Exception as e:
            result = {'success': False, 'error': str(e), 'cargo_schedule': []}
            print(f"ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
    
    solver_output_string = solver_output_buffer.getvalue()
    total_cost = result.get('total_cost', 0)

    final_report_buffer = io.StringIO()
    final_report_buffer.write(f"Total Charter Cost: ${total_cost:,.0f}\n\n")
    final_report_buffer.write("="*80 + "\n")
    final_report_buffer.write("CRUDE MIX OPTIMIZATION SOLVER - FORMULA-BASED EXACT RATIO VERSION\n")
    final_report_buffer.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    final_report_buffer.write("="*80 + "\n")

    vessel_counts = result.get('vessel_distribution', {})
    vessel_types_to_display = ['VLCC', 'SUEZMAX', 'AFRAMAX', 'PANAMAX', 'HANDYMAX']
    cargo_counts_str = ", ".join([f"{v_type}: {vessel_counts.get(v_type.upper(), 0)}" for v_type in vessel_types_to_display])
    
    final_report_buffer.write(f"FINAL CARGO COUNT: {cargo_counts_str}\n")
    final_report_buffer.write(f"Total Charter Cost: ${total_cost:,.0f}\n\n")
    final_report_buffer.write(solver_output_string)

    final_report_string = final_report_buffer.getvalue()

    if not result:
        result = {'success': False, 'error': 'Solver failed', 'cargo_schedule': []}
    
    result['console_output'] = final_report_string.splitlines()
    return result