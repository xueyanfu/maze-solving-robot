%% Tabula Rasa
clearvars;
close all;
clc;

%% Define robot, target, obstacles
% Small maze
robot = [1, 1]; %robot initial position
target = [10, 10]; %target position
max_val = 10; %size of square maze
new_maze = [
    0, 0, 0, 0, 1, 0, 0, 0, 0, 0;
    0, 0, 0, 0, 1, 0, 0, 0, 0, 0;
    0, 0, 1, 0, 1, 0, 1, 1, 1, 0;
    0, 0, 1, 0, 0, 0, 0, 1, 0, 0;
    0, 0, 1, 1, 1, 1, 0, 1, 0, 0;
    0, 0, 0, 0, 0, 1, 0, 1, 0, 0;
    1, 1, 1, 1, 0, 1, 1, 1, 0, 0;
    0, 0, 0, 0, 0, 0, 0, 1, 0, 0;
    0, 0, 1, 1, 1, 1, 0, 0, 0, 0;
    0, 0, 0, 1, 1, 0, 0, 0, 0, 0
];
logical_maze = logical(new_maze);
[row, col] = find(new_maze == 1);
obstacles = [col, max_val - row + 1];


 %Big maze
 %load('maze.mat');
 %maze_flipped = flipud(maze); %post-process maze
 %maze_flipped = maze_flipped'; %post-process maze
 
 %robot = 20*[1, 1]; %robot initial position
 %target = 20*[12, 12]; %target position
 %[obstacles(:,1),obstacles(:,2)] = find(maze_flipped==1); %obstacles vector
 %max_val = 264; %size of square maze8

%% Preprocess variables
xStart = robot(1);
yStart = robot(2);
xTarget = target(1);
yTarget = target(2);

%% Initialise search
OPEN = [];
CLOSED = [];
% structure of lists: xNode, yNode, xParent, yParent, fn

%put all obstacles in CLOSED, parents at 0 (would cause an error if
%picked), fitness at 0 (always better than current fitness, thus will never
%be put into OPEN)
n_obs = size(obstacles,1);
for i = 1:n_obs
    CLOSED(i,:) = [obstacles(i,1), obstacles(i,2), 0, 0, 0];
end

%put start node in OPEN
OPEN(1,:) = [xStart, yStart, xStart, yStart, 0];

%% Start A* Algorithm
while isempty(OPEN) ~= 1
    [sort_OPEN, ind] = sortrows(OPEN,5); %sort according to lowest fn
    q = sort_OPEN(1,:); %pick lowest fn -> parent
    OPEN(ind(1),:) = []; %remove from OPEN
    
    %create children (8-neighborhood around parent node)
    xNode = q(1,1);
    yNode = q(1,2);
    children = [xNode-1, yNode-1; xNode-1, yNode; xNode-1, yNode+1; xNode, yNode+1;...
        xNode+1, yNode+1; xNode+1, yNode; xNode+1, yNode-1; xNode, yNode-1];
    
    out_of_bounds = (children<1 | children>max_val); %check if out of bounds
    children(sum(out_of_bounds,2)>=1,:) = []; %remove out of bounds
    
    for i = 1:size(children,1) %iterate through children
        skip = 0; %reset skip flag
        xChild = children(i,1);
        yChild = children(i,2);
        
        %if child is target, empty OPEN and quit search
        if (xChild == xTarget && yChild == yTarget)
            solution = 1;
            last_parent = q(1,1:2);
            OPEN = [];
            break;
        end
        
        %calculate fitness
        fn = q(5) ...
            + distance(q(1),q(2),xChild,yChild) ...
            + distance(xChild,yChild,xTarget,yTarget);
        
        %skip child if already better version in OPEN
        for j = 1:size(OPEN,1)
            if (xChild == OPEN(j,1) && yChild == OPEN(j,2))
                if fn>OPEN(j,5)
                    skip = 1;
                end
            end
        end
        
        %skip child of already better version in CLOSED
        for k = 1:size(CLOSED,1)
            if (xChild == CLOSED(k,1) && yChild == CLOSED(k,2))
                if fn>CLOSED(k,5)
                    skip = 1;
                end
            end
        end
        
        %otherwise put into OPEN
        if skip == 0
            OPEN(end+1,:) = [xChild, yChild, q(1), q(2), fn];
        end
 
    end
    
    %put parent in CLOSED
    CLOSED(end+1,:) = q;
end
%% Retrace path From Target
if solution == 1
    disp('solution found')
    path = [xTarget, yTarget];
    path(end+1,:) = [last_parent(1) last_parent(2)];
    
    %start from target and retrace backwards through parents
    while (path(end,1) ~= xStart || path(end,2) ~= yStart)
        [a, b] = ismember([path(end,1), path(end,2)], CLOSED(:,1:2),'rows');
        path(end+1,:) = CLOSED(b,3:4);
    end
    path = flipud(path) %flip to get correct order
else
    disp('no solution found')
end

%% Plot Search
% Space=2, Robot=1, Target = 0, Obstacle=-2, Path = -0.5
map = 2*(ones(max_val,max_val));
map(xStart, yStart) = 1; %set robot
map(xTarget, yTarget) = 0; %set target
for i = 1:n_obs %set obstacles
    map(obstacles(i,1),obstacles(i,2)) = -2;
end
for p = 2:(size(path,1)-1) %set path
    map(path(p,1), path(p,2)) = -0.5;
end

figure;
imagesc(map')
set(gca,'YDir','normal')

%% Functions
function dist = distance(x1,y1,x2,y2)
dist=sqrt((x1-x2)^2 + (y1-y2)^2);
end

%% Map show
%resolution = 1; 
%map = binaryOccupancyMap(new_maze, resolution);
%show(map);