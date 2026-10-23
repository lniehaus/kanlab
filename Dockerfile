# Build stage: compile the TypeScript sources into dist/
ARG NODE_VERSION=18
FROM node:${NODE_VERSION}-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
# .env is not copied into the image; compose passes GA_MEASUREMENT_ID in as a build arg
ARG GA_MEASUREMENT_ID=
RUN npm run build

# Runtime stage: serve dist/ with nginx
FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
