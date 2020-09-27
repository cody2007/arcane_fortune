#include <stdlib.h>
#include <stdio.h>
#include "SDL.h"

SDL_Window *window;
SDL_Renderer *renderer;
SDL_Surface *surface;
int done;

void loop() {
	SDL_Event e;
	while (SDL_PollEvent(&e)) {
		// Re-create when window has been resized
		if ((e.type == SDL_WINDOWEVENT) && (e.window.event == SDL_WINDOWEVENT_SIZE_CHANGED)) {
			SDL_DestroyRenderer(renderer);
			
			surface = SDL_GetWindowSurface(window);
			renderer = SDL_CreateSoftwareRenderer(surface);
			// Clear the rendering surface with the specified color
			SDL_SetRenderDrawColor(renderer, 0xFF, 0xFF, 0xFF, 0xFF);
			SDL_RenderClear(renderer);
		}
		
		if (e.type == SDL_QUIT) {
			done = 1;
			return;
		}

		if ((e.type == SDL_KEYDOWN) && (e.key.keysym.sym == SDLK_ESCAPE)) {
			done = 1;
			return;
		}
	}
	
	SDL_UpdateWindowSurface(window);
}

int main(int argc, char *argv[]) {
	SDL_LogSetPriority(SDL_LOG_CATEGORY_APPLICATION, SDL_LOG_PRIORITY_INFO);
	
	if(SDL_Init(SDL_INIT_VIDEO) != 0) {
		SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "SDL_Init fail : %s\n", SDL_GetError());
		return 1;
	}
	
	/* Create window and renderer for given surface */
	window = SDL_CreateWindow("Chess Board", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 640, 480, SDL_WINDOW_RESIZABLE);
	if(!window) {
		SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Window creation fail : %s\n",SDL_GetError());
		return 1;
	}
	
	surface = SDL_GetWindowSurface(window);
	renderer = SDL_CreateSoftwareRenderer(surface);
	if(!renderer) {
		SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Render creation for surface fail : %s\n",SDL_GetError());
		return 1;
	}
	
	//SDL_SetRenderDrawColor(renderer, 0xFF, 0xFF, 0xFF, 0xFF);
	//SDL_RenderClear(renderer);
	
	done = 0;
	while (!done) {loop();}

	SDL_Quit();
	return 0;
}

