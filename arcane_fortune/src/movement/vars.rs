// final move path from update_move_search()
// (derived from Node data structure)
/*pub struct MvPath<'a,'b>{
        unit_ind: usize, // TODO: pointer
        cur_x_start: usize,
        cur_y_start: usize,

        start_coord: usize, // just used as a convience--used by add_neighbors_to_list, generated in update_move_search()
        end_coord: usize, // used for attacking_movable_to

        action: Option<UnitAction>, // None when not moving unit
        path_coords: Vec<usize>, // path_coords[0] is final dest, path_coords[1] is step before dest...
        // path_coords[path_len-1] is the next step from current position
        actions_req: f32,
            
        max_search_depth: usize, // for update_move_search()

        land_movable_to: fn(usize) -> bool,
        // ^ movable_to(map_coord)
        //   civil_movable_to_dest_city_hall(): owner
        //   civil_movable_to_dest_zone(): owner, dest_bldg,
        //      zone_demand_type, zone_dest

        // following are used as parameters for above land movable fns
        owner: &'a Owner,
        dest_bldg: &'b Bldg,
        zone_demand_type: ZoneDemandType,
        zone_dest: ZoneType
}
*/


